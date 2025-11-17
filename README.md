# GNN Decision Boudary

This repository provides a pipeline to analyse the classification behaviour of a GAT, GCN or GraphSAGE model along 
edit paths between graphs from the TUDataset MUTAG. 
Doing so, it uses the repository `pg_gnn_decision_boundary` and precalculated optimal edit operations between graph pairs from it 
to build the edit path graphs.

A GNN model is trained on the MUTAG dataset with k-fold cross validation. We select the best weight state according to 
test accuracy over all folds and epochs. With this model, the edit path graphs are classified. 

The analysis of the GNN classification behaviour along the edit paths between graphs considers:
- the number of classifiction changes (flips) along paths,
- the location of flips (of a certain order) along paths,
- the graph operations triggering flips per sections of the paths.

For analysis, we consider different groups of paths:
- paths between graphs from the same and from different classes, 
- paths between graphs from the training set, between graphs from the test set or between one graph from the training and one 
from the test set (considering the fold yielding the best test accuracy).

A k-fold cross validation training is defined on training sets augmented with labeled path graph. For each fold, gaphs from paths between a graph pair from the training split are added to the training split.We
They are labeled as follows:
- on paths between graphs of the same class with this respective class
- on paths between graphs of different classes with the source graph's class up to a pre-set relative path progress, 
and with the target graph's class afterwards
We calculate test accuracy statistics to compare training stability to the original dataset training.

### Installation

You can use the ``requirements.txt`` to install dependencies. 

To clone the repository, use

```
git clone --recurse-submodules https://github.com/ilkajw/gnn-decision-boundary.git
```
to include the submodule. 

### Usage

In `config.py`, define the settings for analysis and training on path graph-augmented data:

- the model architecture to use in original dataset training and/or augmented dataset training with `MODEL`, set to `GAT`, `GCN` or `GrapgSAGE`
- the model hyperparameters with `MODEL_KWARGS`
- training parameters `K_FOLDS`, `EPOCHS`, `LEARNING_RATE`, `BATCH_SIZE`
- for analysis, whether to measure path progress according to the cost function or number of operations 
with `DISTANCE_MODE` (`edit_step` recommended)
- whether only to consider fully connected path graphs in the analysis with `FULLY_CONNECTED_ONLY`
- whether to only consider paths with correctly classified source and target graph in analysis 
with `CORRECTLY_CLASSIFIED_ONLY` 
- for augmented data training, after what percentage of path progress to change the label for graphs on paths graphs between 
graphs of different class, with `FLIP_AT`

To run all analysis functions, simply run the `run_analyis_pipeline.py` file. With the default settings, path lengths will be calculated for all paths, whether 
you set `correctly_classified_only` to `True` or `False`. Running the pipeline includes the creation of edit path graphs, so 
you can start training on the augmented dataset afterwards.

If you only want to run certain analysis function, run `run_graph_creation_pipeline.py`, then `precalculations.py` 
and afterwards the analysis file you like (`07_xxx.py` to `11_xxx.py`).

If you only want to run the training on augmented training data, run `run_graph_creation_pipeline.py` first. Then, run
`train_model_on_augmented_data.py`.

### Functionality

#### config.py

Here, define
- the TUDataset to use with `DATASET_NAME` (currently only MUTAG is available due to missing precalculated edit paths for other datasets)
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


#### 01_create_path_graphs.py

The file accesses the repo `pg_gnn_decision_boundary`, where precomputed optimal edit operations are stored, and its 
`utils.EditPath.create_edit_path_graphs` method. Path graph sequences for every dataset graph pair in NetworkX and 
PyG format each are created. Doing so, it drops edge attributes due to the implementation of `utils.EditPath.create_edit_path_graphs`. It adds graph attributes on source graph, target graph, optimization iteration, edit step and the edit operation the graph results from. Every sequence is saved to a `.pt` or `.pkl` file, 
respectively. 

#### 02_assign_cumulative_costs.py
Cumulative costs per path graph are reconstructed according to the cost function used for construction and added as graph attributes to the prior created graphs in the PyG sequences.

#### 03_train_model_on_original_dataset.py

The file defines the initial GNN training on the original TUDataset with k-fold cross validation. It saves the 
best weight state according to test accuracy, as well as mean and sample standard deviation over
final test accuracies per fold as baseline measures.

#### 04_classify_original_dataset.py

All graphs from the TUDataset are classified with the trained GNN. Classifications are saved to a JSON file.


#### 05_classify_path_graphs.py
Path graphs from the prior created sequences are classified by the model trained on the original dataset. 
Classifications are stored as graph attributes and a JSON with all graphs, their attributes and classification is created.

#### 06_precalculations.py

A map on distances and the number of operations per path is calculated. Further, a history of flip occurrences per 
path with the triggering edit operation per flip and its path position is computed, one with path positions measured 
according to the cost function, one according to the number of operations.


#### 07_path_length_statistics.py

This script computes and summarizes statistics for path lengths, measured either by cumulative cost or by
the number of operations, depending on the `DISTANCE_MODE` setting in ``config.py``. It reads precomputed path 
lengths from the file specified by ``DISTANCES_PATH`` and optionally filters the paths based on whether their 
source and target graphs are correctly classified, controlled by `CORRECTLY_CLASSIFIED_ONLY`.

#### 08_path_length_statistics_by_num_flips.py

Path lengths statistics are calculated for path groups distinguished by their number of classification flips:
The file reads per-path flip lists from `FLIPS_PATH` and per-path lengths from `DISTANCES_PATH`, 
groups lengths by the number of flips (`k`), and computes summary statistics (mean, median, standard deviation, 
maximum, and count) for each group. The results are saved as a JSON file in `ANALYSIS_DIR` with the naming convention
`<DATASET_NAME>_<MODEL>_path_length_stats_per_num_flips_by_<DISTANCE_MODE>.json`.

#### 09_num_flips_histograms.py

This script builds histograms of the number of classification flips per path for multiple index-set cuts.
It reads per-path flip lists from `FLIPS_PATH` and counts how many paths have 0, 1, 2, ... flips for each index-set cut,
providing both absolute and relative frequencies. Results are saved to a JSON file in `ANALYSIS_DIR` 
with the naming convention `<DATASET_NAME>_<MODEL>_flips_hist_by_<DISTANCE_MODE>.json`.

#### 10_flip_order_distribution.py

This script analyses **where and how often** classification changes (flips) happen along edit paths.
It organizes these flips into ten equal segments (deciles) of the path, based on the path's length or cost. 
For each number of flips (`k`), it calculates:

- **How many flips** occur in each segment of the path.
- **Which operations** (e.g., adding or removing nodes/edges) are most common in each segment.
- **When flips happen** relative to the start and end of the path.

Results are saved as a JSON file.

#### 11_flip_statistics.py

This script analyses and summarizes how often classification flips occur along edit paths for different path groups 
(e.g., training vs. test sets, same vs. different classes)..
It reads the flip data for each path and calculates statistics — such as the average, median, and maximum number of flips 
per group. The results are saved as a JSON file.


Files ``12_plot_num_flips_histogram.py``, ``13_plot_flip_order_distributions.py``, 
``14_plot_flips_distributions_with_ops.py`` and ``15_plot_operation_heatmaps.py`` plot the analysis results and save plots to
``<ANALYSIS_DIR>/plots/``.
