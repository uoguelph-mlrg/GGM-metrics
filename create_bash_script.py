import numpy as np
np.random.seed(0)


num_layers_lst = [2, 3, 4, 5, 6, 7]
hidden_dims_lst = [5, 10, 15, 20, 25, 30, 35, 40]
seeds = list(range(10))
permutation_exps = ['sample-size-random', 'mixing-gen', 'mixing-random', 'rewiring-edges', 'mode-collapse', 'mode-dropping']
datasets = ['community', 'lobster', 'proteins', 'ego', 'grid', 'zinc']
gnns = ['gin', 'gcn', 'gsage', 'gin-no-cat']
statistics = ['degree', 'orbits', 'clustering', 'nspdk', 'wl']
num_architectures = 20

def randomly_select_gnn_architectures():
    def select_random_arch():
        num_layers = np.random.choice(num_layers_lst)
        hidden_dims = np.random.choice(hidden_dims_lst)
        dct = {'num_layers': num_layers, 'hidden_dims': hidden_dims}
        return dct

    tested_combos = []
    for i in range(num_architectures):
        dct = select_random_arch()
        while dct in tested_combos:
            dct = select_random_arch()
        tested_combos.append(dct)
    return tested_combos

def create_commands():
    gnn_architectures = randomly_select_gnn_architectures()
    bash_cmds = ['#!/bin/bash']
    def generate_gnn_commands(gnn_architectures):
        commands = []
        for dataset in datasets:
            for gnn in gnns:
                for dct in gnn_architectures:
                    num_layers = dct['num_layers']
                    hidden_dims = dct['hidden_dims']

                    for seed in seeds:
                        for exp in permutation_exps:
                            command = 'python main.py --seed={} --permutation_type={} --dataset={} '.format(seed, exp, dataset)
                            command += 'gin --num_layers={} --hidden_dim={} '.format(num_layers, hidden_dims) + \
                                       '--feature_extractor={}'.format(gnn)
                            commands += [command]
        return commands

    def generate_mmd_commands():
        commands = []
        for dataset in datasets:
            for statistic in statistics:
                for seed in seeds:
                    for exp in permutation_exps:
                        command = 'python main.py --seed={} --permutation_type={} --dataset={} '.format(seed, exp, dataset)
                        command += 'mmd-structure --statistic={} --is_parallel'.format(statistic)
                        commands += [command]
        return commands

    bash_cmds += generate_gnn_commands(gnn_architectures)
    bash_cmds += generate_mmd_commands()
    return bash_cmds

bash_cmds = create_commands()
open('all_commands.sh', 'w').write('\n'.join(bash_cmds))
