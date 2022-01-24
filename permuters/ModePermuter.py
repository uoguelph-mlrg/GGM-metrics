import numpy as np
import random
from permuters.BasePermuter import BasePermuter
import networkx as nx
from sklearn.cluster import AffinityPropagation
import grakel
from grakel.kernels import WeisfeilerLehman, VertexHistogram


class ModePermuter(BasePermuter):
    """Permuter for the mode collapsing/dropping experiments.

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
    permuted_set : list of DGL graphs
        The permuted set. Graphs in this set are altered to artificially
        drop or collapse modes.
    step_size : int
        The number of clusters to drop or collapse at each iteration.
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
        random.shuffle(reference_set)
        self.reference_set = reference_set[: len(reference_set) // 2]
        self.permuted_set = reference_set[len(reference_set) // 2:]

        n_clusters, cluster_centers_ixs = self.__cluster_graphs(
            reference_set, helper)

        self.__get_permutation_order(
            n_clusters, cluster_centers_ixs, reference_set)

        self.step_size = max(round(n_clusters * helper.args.step_size), 1)

        super().__init__(evaluator, helper)

    def __cluster_graphs(self, reference_set, helper):
        cluster = GraphClustering()
        cluster_labels, n_clusters, cluster_centers_ixs = cluster.get_clusters(
            reference_set, helper.args.dataset)
        helper.logger.info(f'Number of clusters found: {n_clusters}')

        # Label each graph with its cluster
        for g, cluster_label in zip(reference_set, cluster_labels):
            g.cluster_label = cluster_label

        return n_clusters, cluster_centers_ixs

    def __get_permutation_order(
            self, n_clusters, cluster_centers_ixs, reference_set):
        # Get the order in which to drop/collapse clusters
        self.permute_order = list(range(n_clusters))
        tmp = list(zip(self.permute_order, cluster_centers_ixs))
        random.shuffle(tmp)
        self.permute_order, cluster_centers_ixs = zip(*tmp)
        self.cluster_centers = [
            reference_set[center_ix] for center_ix in cluster_centers_ixs]

    def __call__(self, iter):
        [g._reset_cached_info() for g in self.permuted_set + self.reference_set]
        results = self.evaluator.evaluate_all(self.permuted_set,
                                              self.reference_set)

        results['ratio'] = self.get_ratio(iter)
        results['n_clusters'] = len(self.permute_order)
        return results

class GraphClustering():
    """Performs graph clustering."""

    def get_clusters(self, dataset, dataset_name):
        """Clusters the given graphs using WL kernel.

        Parameters
        ----------
        dataset : list of DGL graphs
            The graphs to cluster
        dataset_name : str
            The name of the dataset. ZINC requires slightly different
            functionality.

        Returns
        -------
        res.labels_ : list of int
            The cluster labels of each graph
        num_clusters : int
            The number of clusters found
        res.cluster_centers_indices_ : list of int
            The indices that represent a cluster center (used to index dataset)
        """
        dset = self.__prepare_graphs(dataset)

        res = self.__cluster_graphs(dset, dataset_name)
        num_clusters = max(res.labels_)

        return res.labels_, num_clusters, res.cluster_centers_indices_

    def __prepare_graphs(self, dataset):
        nx_dataset = [  # Convert to nx
            g.cpu().to_networkx().to_undirected() for g in dataset]

        # Add degree attributes
        for g in nx_dataset:
            nx.set_node_attributes(g, dict(g.degree()), 'degree')

        # Convert dataset to grakel and delete nx dataset
        dset = grakel.graph_from_networkx(nx_dataset, node_labels_tag='degree')
        for g in nx_dataset:
            del g

        return dset

    def __cluster_graphs(self, dset, dataset_name):
        gk = WeisfeilerLehman(
            n_iter=1, base_graph_kernel=VertexHistogram, normalize=True)

        K = gk.fit_transform(dset)
        if dataset_name == 'zinc':
            K = K.astype(np.float32)

        clusterer = AffinityPropagation(
            max_iter=1000, copy=False,
            random_state=0)  # Don't randomize clustering

        res = clusterer.fit(K)
        return res

    def experiment_complete(self, iter):
        """Determine if the experiment is complete."""
        return ((iter - 1) * self.step_size) >= len(self.permute_order)

class ModeCollapsePermuter(ModePermuter):
    """Permuter for the mode collapse experiments."""

    def permute_graphs(self, iter):
        """Collapse a set number of clusters.

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
        if self.experiment_complete(iter):
            return False

        clusters_to_collapse_ixs = self.__get_clusters_to_collapse(iter)
        for ix in clusters_to_collapse_ixs:
            cluster_to_collapse = self.permute_order[ix]
            cluster_center = self.cluster_centers[ix]

            # Replace graphs with cluster center
            self.permuted_set = [
                g if g.cluster_label != cluster_to_collapse
                else cluster_center for g in self.permuted_set]

        return True

    def __get_clusters_to_collapse(self, iter):
        lower_ix = (iter - 1) * self.step_size
        upper_ix = min(len(self.permute_order), iter * self.step_size)
        clusters_to_collapse_ixs = range(lower_ix, upper_ix)
        return clusters_to_collapse_ixs

    def get_ratio(self, iter):
        """Get the ratio of collapsed clusters to total # clusters.

        Parameters
        ----------
        iter : int
            The current iteration.

        Returns
        -------
        float :
            The ratio of collapsed clusters to total # clusters.
        """
        return min((iter * self.step_size) / len(self.permute_order), 1)


class ModeDroppingPermuter(ModePermuter):
    """Permuter for the mode dropping experiments."""

    def permute_graphs(self, iter):
        """Drop a set number of clusters.

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
        if self.experiment_complete(iter):
            return False

        clusters_to_drop_ixs = self.__get_clusters_to_drop(iter)
        for ix in clusters_to_drop_ixs:
            cluster_to_drop = self.permute_order[ix]
            replacements = self.__get_replacement_graphs(cluster_to_drop)

            # Remove graphs that are in the cluster to drop, and replace with
            # other graphs in this list
            self.permuted_set = [g if g.cluster_label != cluster_to_drop
                                 else replacement for g, replacement
                                 in zip(self.permuted_set, replacements)]

        return True

    def __get_clusters_to_drop(self, iter):
        lower_ix = (iter - 1) * self.step_size
        upper_ix = min(len(self.permute_order) - 1, iter * self.step_size)
        clusters_to_drop_ixs = range(lower_ix, upper_ix)
        return clusters_to_drop_ixs

    def __get_replacement_graphs(self, cluster_to_drop):
        """Get replacements for the graphs we are deleting from the permuted set."""
        possible_replacements = [
            g for g in self.permuted_set if g.cluster_label != cluster_to_drop]
        replacements_ixs = np.random.choice(
            len(possible_replacements), size=len(self.permuted_set))
        replacements = [possible_replacements[ix] for ix in replacements_ixs]
        return replacements

    def get_ratio(self, iter):
        """Get the ratio of dropped clusters to total # clusters.

        Parameters
        ----------
        iter : int
            The current iteration.

        Returns
        -------
        float :
            The ratio of dropped clusters to total # clusters.
        """
        return min((iter * self.step_size) / (len(self.permute_order) - 1), 1)
