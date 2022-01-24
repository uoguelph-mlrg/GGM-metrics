import numpy as np
import copy
import random
from permuters.BasePermuter import BasePermuter
import torch


class RewiringPermuter(BasePermuter):
    """Permuter for the edge rewiring experiments.

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
    dataset : str
        The name of the dataset.
    permuted_set : list of DGL graphs
        The permuted set of graphs in which edges are rewired.
    step_size : float
        How much the edge rewiring probability increases at each iteration.
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
        self.dataset = args.dataset

        self.reference_set = reference_set

        random.shuffle(self.reference_set)
        self.permuted_set = [copy.deepcopy(g).to(self.device)
                             for g in self.reference_set]

        self.step_size = args.step_size

        super().__init__(evaluator, helper)

    def permute_graphs(self, iter):
        """Rewire the edges of each graph in the permuted set.

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

        # Create a copy of reference set so we can safely rewire edges.
        self.permuted_set = [copy.deepcopy(g).to(self.device)
                             for g in self.reference_set]

        for g in self.permuted_set:
            self.__rewire_edges(g, probability)

        return True

    def __rewire_edges(self, g, probability):
        selected_edges = self.__get_edges_to_rewire(g, probability)
        if len(selected_edges) == 0:
            return

        to_keep, to_disconnect, to_connect = self.__get_nodes(g, selected_edges)

        self.__delete_edges(g, to_keep, to_disconnect, to_connect)

        self.__add_edges(g, to_keep, to_connect)

    def __get_edges_to_rewire(self, g, probability):
        num_edges = len(g.edges()[0])  # This double counts edges in undirected
        selected_edges = np.random.binomial(
            1, probability, size=num_edges // 2).nonzero()[0]
        return selected_edges

    def __get_nodes(self, g, selected_edges):
        to_keep = np.random.binomial(
            1, 0.5, size=len(selected_edges))
        to_disconnect = np.logical_not(to_keep).astype(int)

        edges = torch.tril(
            g.adjacency_matrix().to_dense()).nonzero().transpose(1, 0).numpy()
        to_disconnect = edges[to_disconnect, selected_edges]
        to_keep = edges[to_keep, selected_edges]

        to_connect = self.__get_nodes_to_connect(g, to_keep, to_disconnect)

        assert g.has_edges_between(to_keep, to_disconnect).all()
        assert g.has_edges_between(to_disconnect, to_keep).all()

        return to_keep, to_disconnect, to_connect

    def __get_nodes_to_connect(self, g, to_keep, to_disconnect):
        to_connect = np.random.choice(
            range(g.number_of_nodes()), size=len(to_keep))

        return self.__resample_new_nodes(self, to_keep, to_disconnect, to_connect)

    def __resample_new_nodes(self, g, to_keep, to_disconnect, to_connect):
        """Resample new nodes if we sample the node we are disconnecting or if it
        results in a self-loop."""

        for ix, (kept_node, old_node, new_node) in enumerate(zip(to_keep, to_disconnect, to_connect)):
            if self.dataset != 'ego':
                while new_node == kept_node or new_node == old_node:
                    new_node = np.random.choice(range(g.number_of_nodes()))
            else:
                while new_node == old_node:
                    new_node = np.random.choice(range(g.number_of_nodes()))
            to_connect[ix] = new_node

        return to_connect

    def __delete_edges(self, g, to_keep, to_disconnect, to_connect):
        old_num_edges = g.number_of_edges()

        eids = torch.cat(
            [g.edge_ids(to_keep, to_disconnect),
             g.edge_ids(to_disconnect, to_keep)])
        if 'attr' in g.edata:
            g.deleted_edge_attr = torch.clone(g.edata['attr'][eids])
        else:
            g.deleted_edge_attr = None
        g.remove_edges(eids)
        g._reset_cached_info()

        if self.dataset != 'ego':
            assert g.number_of_edges() == (old_num_edges - len(eids)), f'{g.number_of_edges()}, {old_num_edges}, {len(eids)}'
        assert not g.has_edges_between(to_keep, to_disconnect).all()
        assert not g.has_edges_between(to_disconnect, to_keep).all()

    def __add_edges(self, g, to_keep, to_connect):
        g.add_edges(to_keep, to_connect)
        g.add_edges(to_connect, to_keep)
        g._reset_cached_info()
        if g.deleted_edge_attr is not None:
            g.edata['attr'][-len(eids): ] = g.deleted_edge_attr

        if self.dataset != 'ego':
            assert g.number_of_edges() == old_num_edges, f'{old_num_edges}, {g.number_of_edges()}'

    def __call__(self, iter):
        results = self.evaluator.evaluate_all(self.permuted_set,
                                              self.reference_set)
        results['ratio'] = iter * self.step_size
        return results
