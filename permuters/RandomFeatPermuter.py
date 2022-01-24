import numpy as np
import copy
import random
from permuters.BasePermuter import BasePermuter
import torch


class RandomFeatPermuter(BasePermuter):
    """Permuter for randomizing node or edge features.

    Parameters
    ----------
    reference_set : list of DGL graphs
        The original dataset.
    evaluator : evaluator.Evaluator
        Computes desired metrics
    helper : helper.ExperimentHelper
        Used for experiment logging & tracking

    Attributes
    ----------
    device : torch.device
        The device to move graphs to.
    permuted_set : list of DGL graphs
        The permuted set of graphs in which features are randomized.
    step_size : float
        How much the feature randomization probability increases
        at each iteration.
    """

    def __init__(self, reference_set, evaluator, helper):
        """Initialize experiment and perform first iteration.

        Parameters
        ----------
        reference_set : list of DGL graphs
            The original dataset.
        evaluator : evaluator.Evaluator
            Computes desired metrics
        helper : helper.ExperimentHelper
            Used for experiment logging & tracking
        """
        args = helper.args
        self.device = args.device

        self.reference_set = reference_set
        random.shuffle(self.reference_set)
        self.permuted_set = [copy.deepcopy(g).to(self.device)
                             for g in self.reference_set]

        self.step_size = args.step_size

        if 'randomize-nodes' in helper.args.permutation_type:
            self.__get_features = lambda g: g.ndata['attr']
        else:
            self.__get_features = lambda g: g.edata['attr']

        super().__init__(evaluator, helper)


    def permute_graphs(self, iter):
        """Randomize the features of each graph in the permuted set.

        Parameters
        ----------
        iter : int
            The current iteration

        Returns
        -------
        bool
            Whether the permutation was successful. Returns False if failed
            (and experiment is complete).
        """
        probability = iter * self.step_size
        if probability > 1.0:
            return False

        self.permuted_set = [copy.deepcopy(g).to(self.device)
                             for g in self.reference_set]

        for ix, g in enumerate(self.permuted_set):
            self.__permute_features(g, probability)

        return True

    def __permute_features(self, g, probability):
        feats = self.__get_features(g)
        selected_feats = np.random.binomial(
            1, probability, size=feats.shape[0]).nonzero()[0]
        if len(selected_feats) == 0:
            return

        # Assume features are a single one-hot encoded feat
        num_feats = feats.shape[1]
        one_hot = torch.eye(num_feats).to(feats.device)
        new_feats = np.random.choice(num_feats, size=len(selected_feats))

        feats[selected_feats] = one_hot[new_feats]

    def __call__(self, iter):
        """Compute desired metrics"""
        results = self.evaluator.evaluate_all(self.permuted_set,
                                              self.reference_set)
        results['ratio'] = iter * self.step_size
        return results
