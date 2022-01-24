
This README describes how to use our metrics in your own work.
# Requirements and installation

The main requirements are:
- Python 3.7
- PyTorch 1.8.1
- DGL 0.6.1

To install the required packages:
```
cd evaluation/
pip install -r requirements.txt
```
Following that, install an appropriate version of [DGL 0.6.1](https://www.dgl.ai/pages/start.html) for your system.
Note that the Kernel Distance metric has a Tensorflow dependency that isn't included above. If you'd like to use this metric, also run:
```
pip install tensorflow
pip install tensorflow-gan
```
That's it! View [Evaluation_examples.ipynb](../Evaluation_examples.ipynb) for examples of how to evaluate any graph generative model using our metrics. A portion of the notebook is copied below for posterity:
```
# Generate graphs for demonstration purposes
import utils.graph_generators as gen
import torch
import dgl

grids = gen.make_grid_graphs()
lobsters = gen.make_lobster_graphs()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
grids = [dgl.DGLGraph(g).to(device) for g in grids] # Convert graphs to DGL from NetworkX
lobsters = [dgl.DGLGraph(g).to(device) for g in lobsters] # Convert graphs to DGL from NetworkX
```

```
# Compute all GNN-based metrics at once
from evaluation.evaluator import Evaluator
evaluator = Evaluator(device=device)
evaluator.evaluate_all(grids, lobsters)
```

```
# Alternatively, compute a single GNN-based metric. See evaluation/gin_evaluation.py for other metrics.
from evaluation.gin_evaluation import load_feature_extractor, MMDEvaluation

# Can tweak GIN hyperparameters, however defaults are set to our recommendations
gin = load_feature_extractor(device=device)
# Can tweak hyperparameters of MMD RBF, however defaults are set to our recommendations
mmd_eval = MMDEvaluation(gin)
result, time = mmd_eval.evaluate(grids, lobsters)
print('result: {}, time to compute: {:.3f}s'.format(result, time))
```

