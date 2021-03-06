
# On Evaluation Metrics for Graph Generative Models
Authors: [Rylee Thompson](https://scholar.google.ca/citations?user=pRy6BiAAAAAJ&hl=en), [Boris Knyazev](http://bknyaz.github.io/), [Elahe Ghalebi](https://scholar.google.com/citations?user=h5ZwVzcAAAAJ&hl=en), 
[Jungtaek Kim](https://jungtaek.github.io/), [Graham Taylor](https://www.gwtaylor.ca/)

This is the official repository for the paper [On Evaluation Metrics for Graph Generative Models](https://arxiv.org/abs/2201.09871). Our evaluation metrics enable the efficient computation of the distance between two sets of graphs regardless of domain. In addition, they are more expressive than previous metrics and easily incorporate continuous node and edge features in evaluation. **<font size=6> If you're primarily interested in using our metrics in your work, please see [evaluation/](./evaluation) for a more lightweight setup and installation and [Evaluation_examples.ipynb](./Evaluation_examples.ipynb) for examples on how to utilize our code. </font>** The remainder of this README describes how to recreate our results which introduces additional dependencies.

# Table of Contents  
- [Requirements and installation](#requirements-and-installation)
- [Reproducing main results](#reproducing-main-results)
  * [Permutation experiments](#permutation-experiments)
  * [Pretraining GNNs](#pretraining-gnns)
  * [Generating graphs](#generating-graphs)
- [Visualization](#visualization)
- [License](#license)
- [Citation](#citation)




# Requirements and installation

The main requirements are:
- Python 3.7
- PyTorch 1.8.1
- DGL 0.6.1

```
pip install -r requirements.txt
```
Following that, install an appropriate version of [DGL 0.6.1](https://www.dgl.ai/pages/start.html) for your system and download the proteins and ego datasets by running `./download_datasets.sh`.

# Reproducing main results
The arguments of our scripts are described in [config.py](./config.py). 
## Permutation experiments
Below, examples to run the scripts to run certain experiments are shown. In general, experiments can be run as:
```
python main.py --permutation_type={permutation type} --dataset={dataset}\
{feature_extractor} {feature_extractor_args}
```
For example, to run the mixing random graphs experiment on the proteins dataset using random-GNN-based metrics for a single random seed:
```
python main.py --permutation_type=mixing-random --dataset=proteins\
gnn
```
The hyperparameters of the GNN are set to our recommendations by default, however, they are easily changed by additional flags. To run the same experiment using the degree MMD metric:
```
python main.py --permutation_type=mixing-random --dataset=proteins\
mmd-structure --statistic=degree
```
Rank correlations are automatically computed and printed at the end of each experiment, and results are stored in experiment_results/. Recreating our results requires running variations of the above commands thousands of times. To generate these commands and store them in a bash script automatically, run `python create_bash_script.py`.

## Pretraining GNNs
To pretrain a GNN for use in our permutation experiments, run `python GIN_train.py`, and see [GIN_train.py](./GIN_train.py) for tweakable hyperparameters. Alternatively, the pretrained models used in our experiments can be downloaded by running `./download_pretrained_models.sh`. Once you have a pretrained model, the permutation experiments can be ran using: 
```
python main.py --permutation_type={permutation type} --dataset={dataset}\
gnn --use_pretrained {feature_extractor_args}
```

## Generating graphs
Some of our experiments use graphs generated by [GRAN](https://arxiv.org/abs/1910.00760). To find instructions on training and generating graphs using GRAN, please see the [official GRAN repository](https://github.com/lrjconan/GRAN). Alternatively, the graphs generated by GRAN used in our experiments can be downloaded by running `./download_gran_graphs.sh`.

# Visualization
All code for visualizing results and creating tables is found in [data_visualization.ipynb](./data_visualization.ipynb).

# License

We release our code under the [MIT license](./LICENSE).

# Citation

```
@inproceedings{thompson2022evaluation,
  title={On Evaluation Metrics for Graph Generative Models},
  author={Thompson, Rylee, and Knyazev, Boris and Ghalebi, Elahe and Kim, Jungtaek, and Taylor, Graham W},
booktitle={International Conference on Learning Representations},
  year={2022}  
}
```
