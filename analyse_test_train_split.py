import itertools
import json
import os

import numpy as np
from config import DATASET_NAME
if __name__ == "__main__":

    save_path = f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_changes_per_path_train_vs_test_split.json"

    with open("model/best_split.json") as f:
        split = json.load(f)

    with open(f"data/{DATASET_NAME}/predictions/{DATASET_NAME}_edit_path_predictions.json", "r") as f:
        predictions = json.load(f)

    with open(f"data/{DATASET_NAME}/predictions/{DATASET_NAME}_changes_per_path.json") as f:
        changes_dict = json.load(f)

    # create index sets of train and test graphs
    train_set = set(split["train_idx"])
    test_set = set(split["test_idx"])

    change_counts = {
        'train_train': [],
        'test_test': [],
        'train_test': []
    }

    # calculate number of changes in paths belonging to train_train, test_test, train_test
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
        'train_train': {'num pairs': len(change_counts['train_train']),
                        'mean': np.mean(change_counts['train_train']) if change_counts['train_train'] else 0,
                        'std': np.std(change_counts['train_train']) if change_counts['train_train'] else 0},

        'test_test': {'num pairs': len(change_counts['test_test']) ,
                      'mean': np.mean(change_counts['test_test']) if change_counts['test_test'] else 0,
                      'std': np.std(change_counts['test_test']) if change_counts['test_test'] else 0},

        'train_test': {'num pairs': len(change_counts['train_test']),
                       'mean': np.mean(change_counts['train_test']) if change_counts['train_test'] else 0,
                       'std': np.std(change_counts['train_test']) if change_counts['train_test'] else 0}
    }

    print(stats_changes_per_path)

    # save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(stats_changes_per_path, f, indent=2)