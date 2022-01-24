import dgl
import numpy as np
import torch
from utils import experiment_logging as helper
from evaluation.evaluator import Evaluator
import traceback
import logging
from permuters import SampleSizePermuter, MixingPermuter, RewiringPermuter, ModePermuter, ComputationEffPermuter, RandomFeatPermuter
import utils.graph_generators as generators
import warnings
import time
from config import Config
warnings.filterwarnings('ignore')


def generate_dataset(args, device):
    """Generate (or load) a given dataset.

    Parameters
    ----------
    args : Argparse dict
        The command-line args parsed by Argparse
    device : torch.device
        The device to move the generated graphs to.

    Returns
    -------
    List of DGL graphs
        The generated (or loaded) dataset moved to the
        specified device.

    """
    dataset_name = args.dataset
    seed = args.seed
    if dataset_name == 'grid':
        reference_dset = generators.make_grid_graphs()
    elif dataset_name == 'lobster':
        reference_dset = generators.make_lobster_graphs()
    elif dataset_name == 'er':  # erdos-renyi
        args.er_p = np.random.uniform(low=0.05, high=0.95)
        reference_dset = generators.make_er_graphs(seed, args.er_p)
    elif dataset_name == 'proteins':
        reference_dset = generators.load_proteins()
    elif dataset_name == 'lego':
        reference_dset = generators.load_lego()
    elif dataset_name == 'community':
        reference_dset = generators.make_community_graphs()
    elif dataset_name == 'community-large':
        reference_dset = generators.make_community_graphs_large()
    elif dataset_name == 'ego':
        reference_dset = generators.make_ego_graphs()
    elif dataset_name == 'zinc':
        reference_dset = generators.load_zinc()
    elif dataset_name == 'zinc-large':
        reference_dset = generators.load_zinc_large()
    elif dataset_name == 'cifar10':
        reference_dset = generators.load_cifar10()
    else:
        raise Exception(dataset_name)

    print('Dataset size:', len(reference_dset))
    print('Mean num nodes:', np.mean([g.number_of_nodes() for g in reference_dset]))
    print('Mean num edges:', np.mean([g.number_of_edges() for g in reference_dset]))
    print('Max num nodes:', np.max([g.number_of_nodes() for g in reference_dset]))
    print('Min num nodes:', np.min([g.number_of_nodes() for g in reference_dset]))
    print('Max num edges:', np.max([g.number_of_edges() for g in reference_dset]))
    print('Min num edges:', np.min([g.number_of_edges() for g in reference_dset]))

    if not isinstance(reference_dset[0], dgl.DGLGraph):
        reference_dset = [dgl.DGLGraph(g) for g in reference_dset]
    reference_dset = [g.to(device) for g in reference_dset]

    return reference_dset


def get_graph_permuter(helper, evaluator):
    """Initialize the experiment.

    Parameters
    ----------
    helper : helper.ExperimentHelper
        General experiment helper --- logging results etc.
    evaluator : Evaluator
        The evaluator object used to compute each metric.

    Returns
    -------
    BasePermuter
        The graph permuter that alters the graphs according
        to the specified experiments and computes metrics.

    """
    args = helper.args
    reference_set = generate_dataset(args, device=args.device)
    permutation_type = args.permutation_type

    if permutation_type == 'sample-size-random':
        return SampleSizePermuter.SampleSizePermuter(
            reference_set=reference_set, evaluator=evaluator, helper=helper)

    elif permutation_type == 'mixing-gen'\
            or permutation_type == 'mixing-random':
        return MixingPermuter.MixingPermuter(
            reference_set=reference_set, evaluator=evaluator, helper=helper)

    elif permutation_type == 'rewiring-edges':
        return RewiringPermuter.RewiringPermuter(
            reference_set=reference_set, evaluator=evaluator, helper=helper)

    elif permutation_type == 'mode-collapse':
        return ModePermuter.ModeCollapsePermuter(
            reference_set=reference_set, evaluator=evaluator, helper=helper)

    elif permutation_type == 'mode-dropping':
        return ModePermuter.ModeDroppingPermuter(
            reference_set=reference_set, evaluator=evaluator, helper=helper)

    elif permutation_type == 'computation-eff-size':
        return ComputationEffPermuter.ComputationEffPermuter(
            reference_set=reference_set, evaluator=evaluator,
            helper=helper, type='size')

    elif permutation_type == 'computation-eff-qty':
        return ComputationEffPermuter.ComputationEffPermuter(
            reference_set=reference_set, evaluator=evaluator,
            helper=helper, type='qty')

    elif permutation_type == 'computation-eff-edges':
        return ComputationEffPermuter.ComputationEffPermuter(
            reference_set=reference_set, evaluator=evaluator,
            helper=helper, type='edges')

    elif 'randomize' in permutation_type:
        return RandomFeatPermuter.RandomFeatPermuter(
            reference_set=reference_set, evaluator=evaluator, helper=helper)
    else:
        raise Exception('not implemented')


if __name__ == '__main__':
    # Fetch the command line arguments
    config = Config().parse()

    # For logging results mostly
    helper = helper.ExperimentHelper(
        config, results_dir=config.results_directory)

    try:
        if helper.args.no_cuda or not torch.cuda.is_available():
            helper.args.no_cuda = True
            helper.args.device = torch.device('cpu')
        else:
            helper.args.device = torch.device('cuda')

        # Get object for computing desired metrics
        evaluator = Evaluator(**helper.args)
        # Get object to apply appropriate permutations to graphs
        graph_permuter = get_graph_permuter(helper, evaluator)

        start = time.time()
        graph_permuter.perform_run()
        total = time.time() - start
        total = total / 60
        helper.logger.info('EXPERIMENT TIME: {} mins'.format(total))

    except:
        graph_permuter.save_results_final()
        traceback.print_exc()
        logging.exception('')

    finally:
        helper.end_experiment()
