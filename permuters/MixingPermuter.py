import math
import random
from permuters.BasePermuter import BasePermuter


class MixingPermuter(BasePermuter):
    """Class for the mixing random and mixing generated exps.

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
    reference_set: list of DGL graphs
        See above. Static throughout the experiment.
    permuted_set : list of DGL graphs
        The permuted set. Graphs are slowly deleted from here
        and replaced with random or generated graphs.
    fake_graphs : list of DGL graphs
        The list of random or generated graphs that are added to the
        permuted set.
    step_size : int
        The number of graphs to add/delete at each iteration.
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

        self.reference_set = reference_set
        self.permuted_set = [g for g in self.reference_set]

        self.__get_fake_graphs(args)

        # Compute the number of graphs added/removed at each step
        num_graphs = len(reference_set)
        self.step_size = math.ceil(num_graphs * args.step_size)

        random.shuffle(self.fake_graphs)
        random.shuffle(self.permuted_set)
        # Label graphs so we know how many are real/fake at any point
        for real, fake in zip(self.permuted_set, self.fake_graphs):
            real.is_fake = False
            fake.is_fake = True

        super().__init__(evaluator, helper)

    def __get_fake_graphs(self, args):
        if args.permutation_type == 'mixing-gen':  # Load generated graphs
            self.fake_graphs = self.load_generated_graphs(args)
            self.fake_graphs = self.fake_graphs[: len(self.reference_set)]

        elif args.permutation_type == 'mixing-random':  # Create ER graphs
            self.fake_graphs = self.make_er_graphs(args, self.reference_set)

        else:
            raise Exception('Unimplemented perm type ' + args.permutation_type)

    def permute_graphs(self, iter):
        """Delete graphs from permuted set and replace with fake graphs.

        Parameters
        ----------
        iter : int
            The current iteration.

        Returns
        -------
        bool
            Whether the permutation was successful. Returns False if failed
            (and experiment is complete).
        """
        if self.__experiment_complete():
            return False

        del self.permuted_set[:self.step_size]
        self.permuted_set += self.fake_graphs[:self.step_size]
        del self.fake_graphs[:self.step_size]
        return True

    def __experiment_complete(self):
        return self.get_ratio() >= 1

    def get_ratio(self):
        """Get the ratio of mixed graphs to original graphs.

        Returns
        -------
        float :
            The ratio of mixed graphs in the permuted set.
        """
        num_graphs = len(self.permuted_set)
        fake_graphs = [g for g in self.permuted_set if g.is_fake]
        return len(fake_graphs) / num_graphs

    def __call__(self, iter):
        """Compute metrics at current iter.

        Parameters
        ----------
        iter : int
            The current iteration.

        Returns
        -------
        dict
            Dictionary containing metric values and time for each metric.

        """
        results = self.evaluator.evaluate_all(self.permuted_set,
                                              self.reference_set)

        results['ratio'] = self.get_ratio()
        results['fake_qty'] = results['ratio'] * len(self.reference_set)
        return results
