# GNN Decision Boudary

This repository provides a pipeline to analyse the classification behaviour of a GAT, GCN or GraphSAGE model on 
edit paths between graphs from a TUDataset (currently only MUTAG is available due to mussing precalculations
for other datasets). 
For that, it uses the repository `pg_gnn_decision_boundary` and precalculated optimal edit operations from it 
to build the edit path graphs.

It trains the model on the original dataset with k-fold cross validation and selects the best weight state according to 
test accuracy over all folds and epochs. With the best model, it classifies the edit path graphs. 

It then analyses:
- how many flips happen along the paths,
- where flips (of which order) happen along the paths,
- which operations trigger changes.

For analysis, paths between graphs from the same or from different classes are distinguished, 
as well as paths between graphs from the training set, graphs from the test set or one graph from the training and one 
from the test set.

The repo defines a k-fold cross validation training on training sets augmented with labeled path graphs and
calculates test accuracy statistics to compare training stability to original dataset training.

### Installation

### Usage

In `config.py`, define the settings for analysis and/or training on augmented data:

- the model architecture to use in original dataset training and/or augmented dataset training with `MODEL`
- the model hyperparameters with `MODEL_KWARGS`
- training parameters `K_FOLDS`, `EPOCHS`, `LEARNING_RATE`, `BATCH_SIZE`
- for analysis, whether to measure path progress according to the cost function or number of operations 
with `DISTANCE_MODE`
- whether only to consider fully connected path graph with `FULLY_CONNECTED_ONLY`
- whether to only consider paths with correctly classified source and target graph in analysis 
with `CORRECTLY_CLASSIFIED_ONLY`
- for augmented data training, after what percentage of path progress to change the label for graphs on paths graphs between 
graphs of different class, with `FLIP_AT`

To run all analysis functions, simply run the `run_analyis_pipeline.py` file. With the default setting analysis, path lengths will be calculated for all paths, whether 
you set `correctly_classified_only` to `True` or `False`. Running the pipeline includes graph creation, so 
you can start the training on the augmented dataset afterwards.

If you only want to run certain analysis function, run `run_graph_creation_pipeline.py`, then `precalculations.py` 
and afterwards the analysis file you like (`07_xxx.py` to `11_xxx.py`).

If you only want to run the training on augmented training data, run `run_graph_creation_pipeline.py` first. Then, run
`train_model_on_augmented_data.py`.

### Functionality

#### config.py

Here, define
- the TUDataset to use with `DATASET_NAME` (currently only MUTAG available)
- the model architecture to use in original dataset training and/or augmented dataset training with `MODEL`
- the model hyperparameters with `MODEL_KWARGS`
- training parameters `K_FOLDS`, `EPOCHS`, `LEARNING_RATE`, `BATCH_SIZE`
- for analysis, whether to measure path progress according to the cost function or number of operations 
with `DISTANCE_MODE`
- whether only to consider fully connected path graph with `FULLY_CONNECTED_ONLY`
- whether to only consider paths with correctly classified source and target graph in analysis 
with `CORRECTLY_CLASSIFIED_ONLY`
- for augmented data training, after what percentage of path progress to change the label for graphs on paths graphs between 
graphs of different class, with `FLIP_AT`

#### 01_train_model_on_original_dataset.py

The file defines the initial GNN training on the original TUDataset with k-fold cross validation. It saves the 
best weight state according to test accuracy, as well as mean and sample standard deviation over
final test accuracies per fold as baseline measures.

#### 02_classify_original_dataset.py

All graphs from the TUDataset are classified with the trained GNN and classifications are saved to a JSON, 
to later retrieve which graphs are classified correctly.

#### 03_create_path_graphs.py

The file accesses the repo `pg_gnn_decision_boundary`, where precomputed optimal edit operations are stored, and its 
`utils.EditPath.create_edit_path_graphs` method. The file creates path graph sequences for every pair from the dataset in NetworkX and 
PyG format each. Doing so, it drops edge attributes and adds graph attributes on source graph, target graph, optimization
iteration, edit step and the edit operation the graph results from. Every sequence is saved to a `.pt` or `.pkl` file, 
respectively. 

#### 04_assign_cumulative_costs.py
Here, set the cost function used during edit path calculation. Then, cumulative costs per path graph are reconstructed 
and added as graph attributes to the prior created graphs in the PyG sequences.

#### 05_classify_path_graphs.py
Path graphs from the prior created sequences are defined with the best model trained on the original dataset. 
Predictions are stored as graph attributes and a JSON with all graphs, their attributes and prediction is created.

#### 06_precalculations.py

A map on distances and on the number of operations per path is calculated. Further, a history of flip occurrences per 
path with the triggering edit operation per flip and its path position is computed, one with path positions measured 
according to the cost function, one according to the number of operations.
