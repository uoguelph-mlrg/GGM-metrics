import argparse

class Config():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Experiment args')
        subparsers = self.parser.add_subparsers()

        # General exp. args
        self.parser.add_argument(
            '--no_cuda', action='store_true',
            help='Flag to disable cuda')

        self.parser.add_argument(
            '--seed', default=42, type=int,
            help='the random seed to use')

        self.parser.add_argument(
            '--dataset', default='grid',
            choices=['grid', 'lobster', 'lego', 'proteins',
                     'community', 'ego', 'zinc'],
            help='The dataset to use')

        self.parser.add_argument(
            '--permutation_type', default='sample-size-random',
            choices=['sample-size-random', 'mixing-gen', 'mixing-random',
                     'rewiring-edges', 'mode-collapse', 'mode-dropping',
                     'computation-eff-qty', 'computation-eff-size',
                     'computation-eff-edges', 'randomize-nodes',
                     'randomize-edges'],
            help='The permutation (experiment) to run')

        self.parser.add_argument(
            '--step_size', default=0.01, type=float,
            help='Many experiments have a "step size", e.g. in mixing random\
                graphs, the step size is the percentage (fraction) of random\
                graphs added at each time step.')

        self.parser.add_argument(
            '--results_directory', type=str, default='testing',
            help='Results are saved in experiment_results/{results_directory}')

        gin_parser = subparsers.add_parser('gnn')
        gin_parser.add_argument(
            '--feature_extractor', default='gin',
            choices=['gin', 'gin-no-cat', 'gcn', 'gsage', 'custom'],
            help='The GNN to use. If anything other than "custom" \
            is used, the neighbor_pooling_type, graph_pooling_type, and \
            sometimes the num_mlp_layers parameters are overwritten (i.e.\
            if GIN is selected, neighbor_pooling_type and graph_pooling_type\
            are both overwritten to be "sum").')

        gin_parser.add_argument(
            '--num_layers', default=3, type=int,
            help='The number of prop. rounds in the GNN')

        gin_parser.add_argument(
            '--num_mlp_layers', default=2, type=int,
            help='The number of layers in the MLPs used in graph prop')

        gin_parser.add_argument(
            '--graph_pooling_type', default='sum',
            type=str, help='The method for aggregating node embeddings \
            into graph embedding')

        gin_parser.add_argument(
            '--neighbor_pooling_type', default='sum',
            type=str, help='The method for aggregating neighborhood embeddings\
            into a single node embedding')

        gin_parser.add_argument(
            '--hidden_dim', default=35, type=int,
            help='The node embedding dimensionality. Final graph embed size \
            is hidden_dim * (num_layers - 1)')

        gin_parser.add_argument(
            '--init', default='orthogonal', type=str,
            choices=['default', 'orthogonal'],
            help="The weight init. method for the GNN. Default is PyTorchs\
            default init.")

        gin_parser.add_argument(
            '--use_pretrained', action='store_true',
            help='Flag for using pretrained models. Code looks for model in\
            data/pretrained/{hidden_dim}_{num_layers}_{seed}.h5') #AWESDFASDASDASDASD

        # Parser for the non-GNN-based metrics
        mmd_parser = subparsers.add_parser('mmd-structure')

        mmd_parser.add_argument(
            '--feature_extractor', default='mmd-structure',
            choices=['mmd-structure'])

        mmd_parser.add_argument(
            '--kernel', default='gaussian_emd',
            choices=['gaussian_emd', 'gaussian_rbf'],
            help="The kernel to use for the degree, clustering, orbits, and\
            spectral MMD metrics. Gaussian EMD is the RBF kernel with the L2\
            norm replaced by EMD.")

        mmd_parser.add_argument(
            '--is_parallel', action='store_true',
            help="For degree, clustering, orbits, and spectral MMD metrics.\
            Whether to compute graph statistics in parallel or not.")

        mmd_parser.add_argument(
            '--max_workers', default=4, type=int,
            help="If is_parallel is true, this sets the maximum number of\
            workers.")

        mmd_parser.add_argument(
            '--statistic', default='degree',
            choices=['degree', 'clustering', 'orbits', 'spectral',
                     'WL', 'nspdk'],
            help="The metric to use")

        mmd_parser.add_argument(
            '--sigma', default='single',
            choices=['single', 'range'],
            help="For degree, clustering, orbits, and spectral MMD metrics.\
            Selects whether to use a single sigma (as in GraphRNN and GRAN),\
            or to use the adaptive sigma we propose.")

    def parse(self):
        """Parse the given command line arguments.

        Parses the command line arguments and overwrites
        some values to ensure compatibility.

        Returns
        -------
        Argparse dict: The parsed CL arguments

        """
        args = self.parser.parse_args()
        args.results_directory = '' if args.results_directory is None\
            else args.results_directory + '/'

        args.use_degree_features = True

        if args.dataset == 'zinc':
            args.input_dim = 28  # The number of node features in zinc
            args.edge_feat_dim = 4  # Num edge feats. in zinc
        else:
            args.input_dim = 1  # We use node degree as an int. as node feats.
            args.edge_feat_dim = 0  # No edge features for non-zinc datasets

        assert \
            args.permutation_type not in ['randomize-nodes', 'randomize-edges']\
            or args.dataset in ['zinc']

        if args.use_pretrained:
            args.input_dim = 1  # We use node degree as an int. as node feats.
            args.output_dim = 10  # Number of classes

            args.model_path = f'data/pretrained/{args.hidden_dim}_{args.num_layers}_{args.seed}.h5'

        if 'mmd-structure' not in args.feature_extractor:
            args.results_directory += \
                f'{args.permutation_type}/{args.dataset}/{args.feature_extractor}/{args.graph_pooling_type}'
            if args.feature_extractor != 'gin-no-cat':
                args.graph_embed_size = (args.num_layers - 1) * args.hidden_dim
            else:
                args.graph_embed_size = args.hidden_dim

            if args.feature_extractor == 'gcn':  # Set hyperparams for GCN
                args.num_mlp_layers = 1
                args.neighbor_pooling_type = 'mean'
                args.graph_pooling_type = 'sum'
            elif args.feature_extractor == 'gsage':  # Set hyperparams for SAGE
                args.num_mlp_layers = 1
                args.neighbor_pooling_type = 'max'
                args.graph_pooling_type = 'sum'
            elif 'gin' in args.feature_extractor:  # Set hyperparams for GIN
                args.num_mlp_layers = 2
                args.neighbor_pooling_type = 'sum'
                args.graph_pooling_type = 'sum'

        elif args.feature_extractor == 'mmd-structure'\
                and args.statistic != 'WL' and args.statistic != 'nspdk':
            args.results_directory += \
                f'{args.permutation_type}/{args.dataset}/{args.feature_extractor}/{args.kernel}/{args.statistic}'

        elif args.feature_extractor == 'mmd-structure'\
                and (args.statistic == 'WL' or args.statistic == 'nspdk'):
            args.results_directory += \
                f'{args.permutation_type}/{args.dataset}/{args.feature_extractor}/{args.statistic}'

        return args
