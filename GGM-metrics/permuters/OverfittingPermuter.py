import math
import pickle
from permuters.BasePermuter import BasePermuter
import random
import dgl
import os
import numpy as np
import networkx as nx

class OverfittingPermuter(BasePermuter):
    def __init__(self, reference_set, evaluator, helper):
        args = helper.args
        step_size = args.step_size
        num_graphs = len(reference_set)
        if type(step_size) == float:
            self.step_size = math.ceil(num_graphs * step_size)
        else:
            self.step_size = step_size

        random.shuffle(reference_set)
        if helper.args.overfitting_version == 'original':
            self.train_set = reference_set[: num_graphs // 3]
            self.valid_set = reference_set[num_graphs // 3: num_graphs * 2 // 3]

            self.generated_set = reference_set[num_graphs * 2 // 3:]
            self.generated_set = self.generated_set[:len(self.train_set)]

        elif helper.args.overfitting_version == 'random':
            self.train_set = reference_set[: num_graphs // 2]
            self.valid_set = reference_set[num_graphs // 2:]

            print('use random')
            self.generated_set = self.make_er_graphs(args, reference_set)
            print(self.generated_set[0].device)

        assert len(self.train_set) == len(self.valid_set) and len(self.generated_set) == len(self.valid_set), f'{len(self.train_set)}, {len(self.valid_set)}, {len(self.generated_set)}'
        for train, gen in zip(self.train_set, self.generated_set):
            train.is_train = True
            gen.is_train = False

        super().__init__(evaluator, helper)

    def permute_graphs(self, iter, *args, **kwargs):
        if (self.step_size * iter) > len(self.valid_set):
            return False
        elif (self.step_size * (iter + 1)) >= len(self.valid_set):
            self.generated_set = self.train_set
            return True

        del self.generated_set[:self.step_size]
        self.generated_set += self.train_set[self.step_size * iter: self.step_size * (iter + 1)]
        assert len(self.generated_set) == len(self.train_set)
        return True

    def get_ratio(self):
        """Get the ratio of mixed graphs to original graphs.

        Parameters
        ----------
        permuted_set : list of dgl.DGLGraph
            The permuted set to find the ratio for.

        Returns
        -------
        float :
            The ratio of mixed graphs in the permuted set.
        """
        num_graphs = len(self.generated_set)
        train_graphs = [g for g in self.generated_set if g.is_train]
        return len(train_graphs) / num_graphs

    def __call__(self, iter, *args, **kwargs):
        valid_gen_metrics = self.evaluator.evaluate_all(self.generated_set,
                                                        self.valid_set)

        real_gen_metrics = self.evaluator.evaluate_all(self.generated_set,
                                                       self.train_set)
        final_dict = {}
        for key in real_gen_metrics.keys():
            final_dict['real_gen_' + key] = real_gen_metrics[key]
            final_dict['valid_gen_' + key] = valid_gen_metrics[key]
            final_dict[key] = valid_gen_metrics[key] - real_gen_metrics[key]

        final_dict['ratio'] = self.get_ratio()
        final_dict['copied_qty'] = final_dict['ratio'] * len(self.train_set)

        return final_dict

    def __flip_some_metrics(self):
        print('overwritten sample size')
        results = self.helper.load_results()
        keys_to_flip = ['precision', 'recall', 'density', 'coverage', 'f1_pr', 'f1_dc']
        tmp_dict = {key: [dct[key] for dct in results] for key in results[0].keys() for key_to_flip in keys_to_flip if key_to_flip in key}
        for key in tmp_dict.keys():
            tmp_dict[key] = np.array(tmp_dict[key])
            if 'real_gen' in key or 'valid_gen' in key:
                tmp_dict[key] = max(tmp_dict[key]) - tmp_dict[key]
            else:
                tmp_dict[key] = -tmp_dict[key]

        os.remove(os.path.join(self.helper.run_dir, 'results.h5'))
        for ix, res in enumerate(results):
            for key in tmp_dict.keys():
                res[key] = tmp_dict[key][ix]
        return results

    def get_graphs_to_save(self, iter, *args, **kwargs):
        return {'train_set': self.train_set,
                'valid_set': self.valid_set,
                'generated_set': self.generated_set}
