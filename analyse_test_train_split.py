import itertools
import json
import os

import numpy as np
from config import DATASET_NAME, CORRECTLY_CLASSIFIED_ONLY
from analyse_utils import get_class_changes_per_edit_step, get_class_changes_per_decile

if __name__ == "__main__":

    # todo: change to jupyter notebook?
    #       calculate percentage of steps per decile
    #       encapsulate per path logic in function?

    per_path_save_path = f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_changes_per_path_train_vs_test_split.json"
    per_step_save_path = f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_changes_per_edit_step_train_vs_test_split.json"

    with open("model/best_split.json") as f:
        split = json.load(f)

    with open(f"data/{DATASET_NAME}/predictions/{DATASET_NAME}_edit_path_predictions.json", "r") as f:
        predictions = json.load(f)

    with open(f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_changes_per_path.json") as f:
        changes_dict = json.load(f)

    # create index sets of train and test graphs
    train_set = set(split["train_idx"])
    test_set = set(split["test_idx"])

    if CORRECTLY_CLASSIFIED_ONLY:
        pass
        # todo: filter train_set, test_set for correctly classified source, target.
        #  delete filtering from same/diff class idx set functions

    # sets of indices
    train_train_pairs = {tuple(sorted((i, j))) for i, j in itertools.combinations(train_set, 2)}
    test_test_pairs = {tuple(sorted((i, j))) for i, j in itertools.combinations(test_set, 2)}
    train_test_pairs = {tuple(sorted((i, j))) for i, j in itertools.product(train_set, test_set) if i != j}

    change_counts = {
        'train_train': [],
        'test_test': [],
        'train_test': []
    }

    # -------------------------------- PER PATH ANALYSIS ----------------------------------------------------

    # calculate number of changes in paths belonging to either train_train, test_test, train_test
    for pair_str, steps in changes_dict.items():
        i, j = map(int, pair_str.split(","))

        num_changes = len(steps)

        if i in train_set and j in train_set:
            change_counts['train_train'].append(num_changes)
        elif i in test_set and j in test_set:
            change_counts['test_test'].append(num_changes)
        else:
            change_counts['train_test'].append(num_changes)

    # stats: num data points, mean, std
    stats_changes_per_path = {
        'train_train': {'num_paths': len(change_counts['train_train']),
                        'mean': float(np.mean(change_counts['train_train'])) if change_counts['train_train'] else 0,
                        'std': float(np.std(change_counts['train_train'])) if change_counts['train_train'] else 0},

        'test_test': {'num_paths': len(change_counts['test_test']),
                      'mean': float(np.mean(change_counts['test_test'])) if change_counts['test_test'] else 0,
                      'std': float(np.std(change_counts['test_test'])) if change_counts['test_test'] else 0},

        'train_test': {'num_paths': len(change_counts['train_test']),
                       'mean': float(np.mean(change_counts['train_test'])) if change_counts['train_test'] else 0,
                       'std': float(np.std(change_counts['train_test'])) if change_counts['train_test'] else 0}
    }

    print(f"Statistics - Number of changes per path: \n {stats_changes_per_path}")

    # save
    os.makedirs(os.path.dirname(per_path_save_path), exist_ok=True)
    with open(per_path_save_path, "w") as f:
        json.dump(stats_changes_per_path, f, indent=2)

    # ---------------------------------- PER STEP ANALYSIS -------------------------------------------------------

    get_class_changes_per_edit_step(
        idx_pairs_set=train_train_pairs,
        input_dir=f"data/{DATASET_NAME}/predictions/edit_path_graphs_with_predictions",
        output_dir=f"data/{DATASET_NAME}/analysis",
        output_fname=f"{DATASET_NAME}_changes_per_edit_step_train_train.json")

    get_class_changes_per_edit_step(
        idx_pairs_set=test_test_pairs,
        input_dir=f"data/{DATASET_NAME}/predictions/edit_path_graphs_with_predictions",
        output_dir=f"data/{DATASET_NAME}/analysis",
        output_fname=f"{DATASET_NAME}_changes_per_edit_step_test_test.json")

    changes_per_step_test_test = get_class_changes_per_edit_step(
        idx_pairs_set=test_test_pairs,
        input_dir=f"data/{DATASET_NAME}/predictions/edit_path_graphs_with_predictions",
        output_dir=f"data/{DATASET_NAME}/analysis",
        output_fname=f"{DATASET_NAME}_changes_per_edit_step_train_test.json")

    get_class_changes_per_decile(
        idx_pairs_set=train_train_pairs,
        input_dir=f"data/{DATASET_NAME}/predictions/edit_path_graphs_with_predictions",
        output_dir=f"data/{DATASET_NAME}/analysis",
        output_fname=f"{DATASET_NAME}_changes_per_decile_train_train.json"
    )

    get_class_changes_per_decile(
        idx_pairs_set=test_test_pairs,
        input_dir=f"data/{DATASET_NAME}/predictions/edit_path_graphs_with_predictions",
        output_dir=f"data/{DATASET_NAME}/analysis",
        output_fname=f"{DATASET_NAME}_changes_per_decile_test_test.json"
    )

    get_class_changes_per_decile(
        idx_pairs_set=train_test_pairs,
        input_dir=f"data/{DATASET_NAME}/predictions/edit_path_graphs_with_predictions",
        output_dir=f"data/{DATASET_NAME}/analysis",
        output_fname=f"{DATASET_NAME}_changes_per_decile_train_test.json"
    )
