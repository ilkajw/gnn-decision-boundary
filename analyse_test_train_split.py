import itertools
import json

import numpy as np

if __name__ == "__main__":

    dataset_name = "MUTAG"

    with open("model/best_split.json") as f:
        split = json.load(f)

    with open(f"data/{dataset_name}/predictions/{dataset_name}_edit_path_predictions.json", "r") as f:
        predictions = json.load(f)

    with open(f"data/{dataset_name}/predictions/{dataset_name}_changes_per_path.json") as f:
        changes_dict = json.load(f)

    train_set = set(split["train_idx"])
    test_set = set(split["test_idx"])

    change_counts = {
        'train_train': [],
        'test_test': [],
        'train_test': []
    }

    for pair_str, steps in changes_dict.items():
        i, j = map(int, pair_str.split(","))

        num_changes = len(steps)

        if i in train_set and j in train_set:
            change_counts['train_train'].append(num_changes)
        elif i in test_set and j in test_set:
            change_counts['test_test'].append(num_changes)
        else:
            change_counts['train_test'].append(num_changes)

    stats_num_changes = {
        'train_train': {'num pairs': len(change_counts['train_train']),
                        'mean': np.mean(change_counts['train_train']),
                        'std': np.std(change_counts['train_train'])},

        'test_test': {'num pairs': len(change_counts['test_test']),
                      'mean': np.mean(change_counts['test_test']),
                      'std': np.std(change_counts['test_test'])},

        'test_train': {'num pairs': len(change_counts['test_train']),
                       'mean': np.mean(change_counts['test_train']),
                       'std': np.std(change_counts['test_train'])}
    }

    # todo: save, test