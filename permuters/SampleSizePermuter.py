import math
from permuters.BasePermuter import BasePermuter
import random
import os
import numpy as np


class SampleSizePermuter(BasePermuter):
    def __init__(self, reference_set, evaluator, helper):
        args = helper.args
        device = args.device

        num_graphs = len(reference_set)
        random.shuffle(reference_set)
        self.train_set = reference_set[: num_graphs // 2]
        self.val_set = reference_set[num_graphs // 2:]

        self.gen_set = self.make_er_graphs(args, reference_set)
        random.shuffle(self.gen_set)
        self.gen_set = self.gen_set[: num_graphs // 2]
        self.gen_set = [g.to(device) for g in self.gen_set]

        step_size = args.step_size
        self.step_size = math.ceil(num_graphs * step_size)

        self.end_experiment = False

        super().__init__(evaluator, helper)

    def permute_graphs(self, iter, *args, **kwargs):
        end_ix = self.get_end_ix(iter)
        if end_ix > len(self.train_set) and self.end_experiment:
            return False
        elif end_ix > len(self.train_set):
            self.end_experiment = True

        return True

    def get_end_ix(self, iter):
        return (iter * self.step_size) + 7

    def get_subsets(self, end_ix):
        train_set = self.train_set[:end_ix]
        val_set = self.val_set[:end_ix]
        gen_set = self.gen_set[:end_ix]

        return train_set, val_set, gen_set

    def __call__(self, iter, *args, **kwargs):
        end_ix = self.get_end_ix(iter)
        train_set, val_set, gen_set = self.get_subsets(end_ix)
        real_metrics = self.evaluator.evaluate_all(val_set,
                                                   train_set)

        generated_metrics = self.evaluator.evaluate_all(gen_set,
                                                        train_set)
        final_dict = {}
        for key in real_metrics.keys():
            final_dict['real_' + key] = real_metrics[key]
            final_dict['fake_' + key] = generated_metrics[key]
            final_dict[key] = generated_metrics[key] - real_metrics[key]
        final_dict['ratio'] = min(end_ix, len(self.train_set)) / len(self.train_set)
        final_dict['sample_size'] = end_ix

        return final_dict

    def __flip_some_metrics(self):
        results = self.helper.load_results()
        keys_to_flip = ['precision', 'recall', 'density', 'coverage', 'f1_pr', 'f1_dc']
        tmp_dict = {key: [dct[key] for dct in results] for key in results[0].keys() for key_to_flip in keys_to_flip if key_to_flip in key}
        for key in tmp_dict.keys():
            tmp_dict[key] = np.array(tmp_dict[key])
            if 'real' in key or 'fake' in key:
                tmp_dict[key] = max(tmp_dict[key]) - tmp_dict[key]
            else:
                tmp_dict[key] = -tmp_dict[key]

        os.remove(os.path.join(self.helper.run_dir, 'results.h5'))
        for ix, res in enumerate(results):
            for key in tmp_dict.keys():
                res[key] = tmp_dict[key][ix]
        return results
