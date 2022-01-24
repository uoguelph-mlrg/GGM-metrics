import numpy as np
from permuters.BasePermuter import BasePermuter
import networkx as nx
import dgl
import torch

class ComputationEffPermuter(BasePermuter):
    """Code for the computation efficiency experiments.

    Parameters
    ----------
    reference_set : list of DGL graphs
        Used in the experiment where we increase qty. of graphs
        to generate ER graphs that resemble this set.
    evaluator : Evaluator
        Used to compute (and time) metrics
    helper : helper.ExperimentHelper
        Description of parameter `helper`.
    type : str
        The experiment type, i.e. 'qty', 'edges', or 'size'.

    Attributes
    ----------
    device : torch.device
        The device graphs are stored on.
    graph_sizes : list of int
        The graph size at each iteration. I.e., graph_sizes[1]
        contains the size of each graph at the first iteration.
    num_graphs : list of int
        The number of graphs at each iteration, similar to graph_sizes.
    p : list of float
        The sparsity (p parameter) of the ER graphs at each iteration,
        similar to graph_sizes and num_graphs.
    name : str
        The experiment type, i.e. 'qty', 'edges', or 'size'.
    permuted_set : list of DGL graphs
        A list of generated ER graphs. Contains num_graphs[i] graphs,
        each of size graph_sizes[i], with sparsity (p) p[i].

    """
    def __init__(self, reference_set, evaluator, helper, type):
        self.device = helper.args.device

        if type == 'size':
            # This experiment varies the size of each graph in the list
            self.graph_sizes = np.arange(10000, 110000, 10000).tolist()
            self.graph_sizes.insert(0, 1000)
            self.graph_sizes = np.array(self.graph_sizes)

            # Num graphs and num edges (relatively) constant throughout
            self.num_graphs = [50] * len(self.graph_sizes)
            self.p = 10000 / (self.graph_sizes ** 2)  # Approx 10000 edges per graph

        elif type == 'qty':
            # This experiment varies the qty. of graphs
            self.num_graphs = np.arange(1000, 11000, 1000).tolist()
            self.num_graphs.insert(0, 100)

            # Graph size and p constant throughout
            self.graph_sizes = [50] * len(self.num_graphs)
            self.p = [
                g.number_of_edges() / (g.number_of_nodes() ** 2)
                for g in reference_set]
            self.p = [np.mean(self.p)] * len(self.num_graphs)

        elif type == 'edges':
            # This experiment varies the p-parameter --- i.e. number of edges
            self.p = np.arange(0.1, 1.1, 0.1).tolist()
            self.p.insert(0, 0.01)

            # Graph size and num graphs constant throughout
            self.graph_sizes = [1000] * len(self.p) # Max out at ~1m edges
            self.num_graphs = [50] * len(self.p)

        self.name = type

        # Prepare first (or zeroth) iteration
        graph_sizes = self.graph_sizes[0]
        num_graphs = self.num_graphs[0]
        p = self.p[0]
        self.permuted_set = [
            nx.erdos_renyi_graph(graph_sizes, p, seed=ix)
            for ix in range(num_graphs)]
        self.permuted_set = [
            dgl.DGLGraph(g).to(self.device) for g in self.permuted_set]

        super().__init__(evaluator, helper)

    def permute_graphs(self, iter):
        """Generate graphs with specific size, sparsity, and qty for given iter.

        Parameters
        ----------
        iter : int
            The current iteration.

        Returns
        -------
        bool
            Whether the permutation was successful. If false, experiment is complete.
        """
        if iter >= len(self.num_graphs):
            return False

        # Get information for this iteration
        num_graphs = self.num_graphs[iter]
        graph_sizes = self.graph_sizes[iter]
        p = self.p[iter]

        # Delete old graphs to save memory
        self.__clear_old_memory()

        temp = self.__generate_er_graphs(graph_sizes, p, num_graphs)

        self.permuted_set = [dgl.DGLGraph(g).to(self.device) for g in temp]

        # Delete nx graphs as we created a DGL copy.
        for g in temp:
            del g
        del temp

        return True

    def __call__(self, iter, *args, **kwargs):
        """Compute (and time) each metric.

        Parameters
        ----------
        iter : int
            The given iteration.
        *args, **kwargs unused

        Returns
        -------
        dict
            Dictionary containing results for this iteration.
            Stores the time to compute each metric, info regarding
            current stage of the experiment (i.e. graph sizes).

        """
        metrics = self.evaluator.evaluate_all(self.permuted_set,
                                              self.permuted_set)

        for key in list(metrics.keys()):
            if 'time' not in key:
                del metrics[key]
        metrics['num_graphs'] = self.num_graphs[iter]
        metrics['graph_size'] = self.graph_sizes[iter]
        metrics['er_p'] = self.p[iter]
        return metrics

    def __generate_er_graphs(self, graph_sizes, p, num_graphs):
        if self.name == 'edges':
            temp = [
                nx.erdos_renyi_graph(graph_sizes, p, seed=ix)
                for ix in range(num_graphs)]
        else:
            temp = [  # This function is much faster for sparser graphs
                nx.fast_gnp_random_graph(graph_sizes, p, seed=ix)
                for ix in range(num_graphs)]
        return temp

    def __clear_old_memory(self):
        for g in self.permuted_set:
            del g
        del self.permuted_set
        torch.cuda.empty_cache()

    def save_results_final(self):
        """Override method to avoid computing rank corr."""
        pass  # Don't do the rank correlation and stuff that others do
