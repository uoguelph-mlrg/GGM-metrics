import pickle
import logging
import json
import threading
import psutil
import logging
import traceback
from datetime import datetime
import random
import string
import subprocess
import sys
import os
import torch
import copy
import time


class ExperimentHelper():
	import pickle
	import logging
	import json
	import threading
	import psutil
	import logging
	import traceback
	from datetime import datetime
	import random
	import string
	import subprocess
	import sys
	import os
	import torch
	import copy
	"""This class helps to keep track of experiment results.
	It launches a separate thread to track memory usage, and creates and saves
	results to a unique directory it creates"""
	def __init__(self, args, results_dir='gran', train=True):
		"""
		Sets up the experiment results directory. Saves the parameters of the
		experiment to a file (run_dir/config_save.json), launches a memory logger in a
		separate thread (results saved in run_dir/memory.h5). Also assigns a unique
		10 character ID to each run, which can be used for loading config/hyperparameters settings
		from a previous run, or resuming from a previous run.

		Parameters
		----------
		args : dict
			A dictionary containing the hyperparameter/config for the current run.

		results_dir: str
			This class will create the run directory in experiment_results/results_dir.

		train: bool
			In my experiments args contains keys 'config_from' and 'resume_from',
			which optionally contain run IDs we want to reuse the configuration for or
			resume from. This is basically if you want to scan experiment_results/ for
			these IDs to load those configurations or not.
		"""
		if not isinstance(args, dotdict):
			try:
				args = dotdict(vars(args))
			except:
				args = dotdict(args)

		self.__base_dir = os.path.join(os.getcwd(), 'experiment_results')
		if not os.path.exists(self.__base_dir):
			os.mkdir(self.__base_dir)
		self.__results_dir = os.path.join(self.__base_dir, results_dir)
		self.__config_file_name = 'config_save.json'; self.__results_file_name = 'results.h5'
		self.__train = train

		# Create run directory for results
		self.run_dir = self.__make_run_dir(args)
		# Setup the output logger - saved to run_dir/logfile.log
		self.__setup_logger()

		# Setup args (load previous config if necessary)
		if self.__train:
			args = self.__setup_args_train(args)
		# Store the git commit
		try:
			args.git_commit = subprocess.check_output(["git", "describe", '--always']).strip().decode('UTF-8')
		except:
			pass
		self.args = args

		self.logger.info(f'saving results to: {self.run_dir}')
		self.__config_file_name = os.path.join(self.run_dir, self.__config_file_name)
		self.__results_file_name = os.path.join(self.run_dir, self.__results_file_name)

		# Setup memory logger
		self.__memory_logger = MemoryLogger(os.getpid(), self.run_dir)
		self.__memory_logger.start()

		# Create config file
		with open(self.__config_file_name, 'w') as f:
			print(self.args)
			json.dump(self.args, f, indent=4, sort_keys=True)


	def end_experiment(self):
		# Kill the memory logger thread
		self.__memory_logger.join()


	def save_results(self, dict_res):
		"""
		Appends the results from the given epoch to run_dir/results.h5.

		Parameters
		----------
		dict_res : dict
			A dictionary containing the results for the given epoch. Appends the dictionary to a pickle file.
			Expected as dict_res[metric] = value
		"""
		with open(self.__results_file_name, 'ab+') as f:
			pickle.dump(dict_res, f)

	def load_results(self):
		return get_results(run_dir=self.run_dir)


	def checkpoint_model_if_best(self, dict_res, model, optimizer, scheduler, criteria, objective):
		"""
		Checkpoints the given model, optimizer, and scheduler if it has the best performance for the
		given criteria and objective. Saves the model to run_dir/best_model_epoch_(epoch#).h5.

		Parameters
		----------
		criteria: str
			The key in dict_res (from save_results) you want to maximize/minimize (e.g 'train_loss').
		objective: str
			Either 'maximize' or 'minimze': whether you want to maximize or minimize the above criteria.

		Returns: bool
		Whether the model was checkpointed or not (i.e if it was the best one)
		"""
		assert objective == 'maximize' or objective == 'minimize', 'invalid objective, got {}'.format(objective)
		if not hasattr(self, 'best_score'):
			self.best_score = float('inf') if objective == 'minimize' else -float('inf')
			self.new_score_is_better = self.__new_less_than if objective == 'minimize' else self.__new_greater_than

		new_score = dict_res[criteria]
		if self.new_score_is_better(new_score):
			self.best_score = new_score
			prev_best_file = [file for file in os.listdir(self.run_dir) if 'best_model' in file]
			if len(prev_best_file) > 0:
				os.remove(os.path.join(self.run_dir, prev_best_file[0]))
			self.checkpoint_model(model, optimizer, scheduler, best=True)
			return True
		return False

	def checkpoint_model(self, model, optimizer, scheduler, best=False):
		"""
		Checkpoints the given model/optimizer/scheduler to run_dir/checkpoints
		Uses the scheduler to obtain the current epoch.
		"""
		epoch = scheduler.state_dict()['last_epoch']
		filename = os.path.join(self.run_dir, 'models', 'epoch_{:04d}.h5'.format(epoch)) if not best else \
			os.path.join(self.run_dir, 'best_model_epoch_{:04d}.h5'.format(epoch))
		to_save = {'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					'scheduler_state_dict': scheduler.state_dict()}

		torch.save(to_save, filename)


	def __new_less_than(self, new):
		return new < self.best_score

	def __new_greater_than(self, new):
		return new > self.best_score


	def __setup_args_train(self, args):
		config_from = copy.deepcopy(args.get('config_from', None))
		resume_from = copy.deepcopy(args.get('resume_from', None))
		if config_from != None or resume_from != None:
			args = self.__load_previous_args(config_from, resume_from)

		return args


	def __load_previous_args(self, config_from, resume_from):
		"""
		Loads the arguments used in the ID contained in config_from or resume_from.
		Overwrites values with anything passed to the command line (i.e. if you run
		python train.py --learning_rate=0.1, the learning_rate for this experiment will
		be 0.1, regardless of what is in the previous config).
		"""
		resume_from = resume_from if resume_from != None else '------'
		config_from = config_from if config_from != None else '------'
		for root, subdir, files in os.walk(self.__base_dir):
			if resume_from in str(root) or config_from in str(root) and 'results.h5' in files:
				config_file = os.path.join(root, self.__config_file_name)
				self.logger.info(f'Loading config from {config_file}')
				with open(config_file, 'r') as f:
					loaded_config = dotdict(json.load(f))
					argv = ' '.join(sys.argv)
					for ix, arg in enumerate(argv.split('--')):
						if arg == '--' or '.py' in arg:
							continue
						elif '=' in arg:
							arg = arg.strip().split('=')
							option = arg[0]
							value = arg[-1]
						else:
							arg = arg.strip().split(' ')
							option = arg[0]
							value = arg[-1]
						previous_value = loaded_config.get(option, 'empty')
						self.logger.info(f'Updating {option} with {value} (was {previous_value})')
						try:
							loaded_config[option] = eval(value)
						except:
							loaded_config[option] = value
				return loaded_config

		run_id = resume_from if resume_from != None and resume_from != '------' else config_from
		raise Exception('Couldnt find run with id {}'.format(run_id))


	def __setup_logger(self):
		logger = logging.getLogger()

		logger.handlers = []
		handler = logging.StreamHandler(sys.stdout)
		handler.setLevel(logging.DEBUG)
		logger.addHandler(handler)

		logger.setLevel(logging.DEBUG)

		log_format = logging.Formatter('%(message)s')
		filename = os.path.join(self.run_dir, 'logfile.log')
		log_handler = logging.FileHandler(filename)
		log_handler.setLevel(logging.DEBUG)
		log_handler.setFormatter(log_format)

		logger.addHandler(log_handler)
		logger.propagate = False

		self.logger = logging.getLogger()


	def __make_run_dir(self, args):
		resume_from = args.get('resume_from', None)
		if not self.__train or resume_from is None:
			return self.__create_new_run_dir()

		else:
			return find_run_dir(run_id=resume_from)
			# for root, subdir, file in os.walk(self.old_run_dir):
			# 	if resume_from in root:
			# 		return root

		raise Exception('Couldnt return run directory with id {}'.format(resume_from))


	def __create_new_run_dir(self):
		results_path = self.__results_dir.split('/')
		path = '/'
		for subdir in results_path:
			path = path + subdir + '/'
			os.makedirs(path, exist_ok=True)


		# Create random run id
		letters = string.ascii_lowercase + string.ascii_uppercase + string.digits
		run_id = ''.join(random.choice(letters) for i in range(10))
		now = datetime.now()
		# Get current date and time in a string and append run id to get the run directory
		now = now.strftime("%Y%m%d-%H%M")
		now += '-' + run_id
		run_dir = os.path.join(self.__results_dir, now)
		os.mkdir(run_dir)
		os.mkdir(os.path.join(run_dir, 'models'))

		return run_dir


def find_run_dir(run_id, base_dir='experiment_results'):
	for root, subdir, files in os.walk(base_dir):
		if run_id in str(root) and 'memory.h5' in files:
			return root

	raise Exception('Could not find run dir from ids {} in {}'.format(run_id, base_dir))


def get_config(run_id=None, run_dir=None, base_dir='experiment_results'):
	if run_dir is None and run_id is None:
		raise Exception('run_dir and run_id are both None')
	elif run_dir is None:
		run_dir = find_run_dir(run_id, base_dir=base_dir)

	with open(os.path.join(run_dir, 'config_save.json'), 'r') as f:
		config = json.load(f)
	return config


def get_results(run_id=None, run_dir=None, base_dir='experiment_results'):
	if run_dir is None and run_id is None:
		raise Exception('run_dir and run_id are both None')
	elif run_dir is None:
		run_dir = find_run_dir(run_id, base_dir=base_dir)

	with open(os.path.join(run_dir, 'results.h5'), 'rb') as f:
		res = []
		while 1:
			try:
				res += [pickle.load(f)]
			except EOFError:
				break
	return res


def get_memory_usage(run_id=None, run_dir=None, base_dir='experiment_results'):
	if run_dir is None and run_id is None:
		raise Exception('run_dir and run_id are both None')
	elif run_dir is None:
		run_dir = find_run_dir(run_id, base_dir=base_dir)

	with open(os.path.join(run_dir, 'memory.h5'), 'rb') as f:
		res = []
		while 1:
			try:
				res += pickle.load(f)
			except EOFError:
				break
	return res


def get_checkpoint(run_id=None, run_dir=None, base_dir='experiment_results', checkpoint='most recent'):
	if run_dir is None and run_id is None:
		raise Exception('run_dir and run_id are both None')
	elif run_dir is None:
		run_dir = find_run_dir(run_id, base_dir=base_dir)

	if checkpoint == 'most recent' or type(checkpoint) == int:
		model_path = os.path.join(run_dir, 'models')
		if not os.path.exists(model_path):
			raise Exception('Run dir {} has no checkpointed models (models/ folder)'.format(run_dir))
		models = os.listdir(model_path)
		if checkpoint == 'most recent':
			model = sorted(models)[-1]
		else:
			model = [file for file in models if str(checkpoint) in file][0]
		print(f'loading checkpoint from {os.path.join(run_dir, model)}')
		model_path = os.path.join(model_path, model)
		model = torch.load(model_path)

	elif checkpoint == 'best':
		model = [file for file in os.listdir(run_dir) if 'best_model' in file]
		assert len(model) > 0, 'Could not find best model in {}'.format(run_dir)

	else:
		raise Exception('Unsupported argument for checkpoint {}'.format(checkpoint))

	return model

def clear_wandb_info(dir):
	for root, subdir, files in os.walk(dir):
		if 'memory.h5' in files:
			try:
				os.remove(os.path.join(root, 'wandb_info.json'))
			except:
				pass

def remove_run(dir):
	import shutil
	for root, subdir, files in os.walk(dir):
		if 'memory.h5' in files:
			config = json.load(open(os.path.join(root, 'config_save.json'), 'r'))
			if config['seed'] == 7 and 'protein' in root:
				shutil.rmtree(root)
				print('removing ', root)

def tar_old_runs(dir, out_file, remove_compressed_files=True):
    import tarfile
    import shutil
    with tarfile.open(out_file, 'w:gz', compresslevel=6) as tar:
        for root, subdir, files in os.walk(dir):
            if 'memory.h5' in files:
                run_dir = root.split('/')[-1]
                time_str = '-'.join(run_dir.split('-')[: -1])

                if 'nspdk' not in root:
                    shutil.rmtree(root)
                #date = datetime.strptime(time_str, "%Y%m%d-%H%M")
                #breakoff = datetime.strptime('Nov 1 2021', '%b %d %Y')
                #if date > breakoff:
                    #print(root)
                    #tar.add(root)
                    #shutil.rmtree(root)

def find_newest_run(dir):
    # import tarfile
    # import shutil
    # with tarfile.open(out_file, 'w:gz', compresslevel=6) as tar:
    newest = datetime.strptime('Sep 3 1990', '%b %d %Y')
    for root, subdir, files in os.walk(dir):
        if 'memory.h5' in files:
            run_dir = root.split('/')[-1]
            time_str = '-'.join(run_dir.split('-')[: -1])

            date = datetime.strptime(time_str, "%Y%m%d-%H%M")
            if date > newest:
                newest = date
    print(newest)


class dotdict(dict):
	# https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
	"""dot.notation access to dictionary attributes"""
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		for key in self.keys():
			if isinstance(self[key], dict):
				self[key] = dotdict(self[key])



class MemoryLogger():
	def __init__(self, pid, save_dir):
		"""
		pid: The pid of the process whose memory we want to track.
		save_dir: The run directory of the experiment. Results saved to
		save_dir/memory.h5
		"""
		self.pid = pid; self.save_dir = save_dir
		self.finish = False


	def start(self):
		# Launch separate thread
		self.__memory_logger = threading.Thread(target = self.__log_memory_usage, name='mem')
		self.__memory_logger.start()


	def __log_memory_usage(self):
		res_file = os.path.join(self.save_dir, 'memory.h5') #The file to save to
		with open(res_file, 'wb') as f: # Create memory file
			pass
		pid_info = psutil.Process(self.pid)
		mem_usage = []
		while not self.finish: # While the experiment isn't done
			# Put the thread to sleep for given # seconds - I typically use 5
			time.sleep(5)
			mem_usage.append(pid_info.memory_info()[0] / 1e9) # Get memory usage in GB
			if len(mem_usage) >= 120 or self.finish:
				# Save results to file and clear list to save memory
				with open(res_file, 'ab+') as f:
					pickle.dump(mem_usage, f)
				del mem_usage
				mem_usage = []


	def join(self):
		# Kill the thread
		self.finish = True
		self.__memory_logger.join()


class Printer():
	def __init__(self, keys=['loss'], epochs=1):
		self.keys = keys
		self.logger = logging.getLogger()
		self.epochs = epochs

	def print(self, epoch, results):
		str = f'epoch: {epoch}/{self.epochs} '
		for key in self.keys:
			try:
				str += f'{key}: {round(results[key], 7)} '
			except:
				pass

		self.logger.info(str)


def print_log(*args):
	"""Save the given string(s) to the logfile and print to stdout
	Should accept arguments the same as print() does"""
	string = ''
	for arg in args:
		string += str(arg)
	print(string)
	logging.debug(string)
