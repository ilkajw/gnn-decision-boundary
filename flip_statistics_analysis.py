import itertools
import json

import numpy as np

from analyse_utils import get_num_changes_all_paths
from config import DATASET_NAME, CORRECTLY_CLASSIFIED_ONLY
from index_sets_utils import graph_index_pairs_diff_class, graph_index_pairs_same_class

if __name__ == "__main__":

    # todo: construct sets with helpers from index_sets_utils

    # ---------------------------- create index sets ------------------------------

    with open("model/best_split.json") as f:
        split = json.load(f)

    with open(f"data/{DATASET_NAME}/predictions/{DATASET_NAME}_edit_path_predictions.json", "r") as f:
        predictions = json.load(f)

    with open(f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_changes_per_path.json") as f:
        changes_dict = json.load(f)

    # train vs. test split

    train_set = set(split["train_idx"])
    test_set = set(split["test_idx"])

    train_train_pairs = {tuple(sorted((i, j))) for i, j in itertools.combinations(train_set, 2)}
    test_test_pairs = {tuple(sorted((i, j))) for i, j in itertools.combinations(test_set, 2)}
    train_test_pairs = {tuple(sorted((i, j))) for i, j in itertools.product(train_set, test_set) if i != j}

    # same vs. diff class

    diff_class_pairs = graph_index_pairs_diff_class(
        dataset_name=DATASET_NAME,
        correctly_classified_only=CORRECTLY_CLASSIFIED_ONLY,
        save_path=f"data/{DATASET_NAME}/index_sets/{DATASET_NAME}_idx_pairs_diff_class.json")

    same_class_pairs, same_class_0_pairs, same_class_1_pairs = graph_index_pairs_same_class(
        dataset_name=DATASET_NAME,
        correctly_classified_only=CORRECTLY_CLASSIFIED_ONLY,
        save_dir=f"data/{DATASET_NAME}/index_sets/{DATASET_NAME}_idx_pairs")

    # filter train-train, test-test, train-test for same vs diff class paths

    train_train_same = {pair for pair in train_train_pairs if pair in same_class_pairs}
    train_train_same_0 = {pair for pair in train_train_pairs if pair in same_class_0_pairs}
    train_train_same_1 = {pair for pair in train_train_pairs if pair in same_class_1_pairs}
    train_train_diff = {pair for pair in train_train_pairs if pair in diff_class_pairs}

    test_test_same = {pair for pair in test_test_pairs if pair in same_class_pairs}
    test_test_same_0 = {pair for pair in test_test_pairs if pair in same_class_0_pairs}
    test_test_same_1 = {pair for pair in test_test_pairs if pair in same_class_1_pairs}
    test_test_diff = {pair for pair in test_test_pairs if pair in diff_class_pairs}

    train_test_same = {pair for pair in train_test_pairs if pair in same_class_pairs}
    train_test_same_0 = {pair for pair in train_test_pairs if pair in same_class_0_pairs}
    train_test_same_1 = {pair for pair in train_test_pairs if pair in same_class_1_pairs}
    train_test_diff = {pair for pair in train_test_pairs if pair in diff_class_pairs}

    # -------------------- count flips per index set -----------------------------------------

    train_train_same_class_flips = get_num_changes_all_paths(train_train_same, changes_dict)
    train_train_same_class_0_flips = get_num_changes_all_paths(train_train_same_0, changes_dict)
    train_train_same_class_1_flips = get_num_changes_all_paths(train_train_same_1, changes_dict)
    train_train_diff_class_flips = get_num_changes_all_paths(train_train_diff, changes_dict)

    test_test_same_class_flips = get_num_changes_all_paths(test_test_same, changes_dict)
    test_test_same_class_0_flips = get_num_changes_all_paths(test_test_same_0, changes_dict)
    test_test_same_class_1_flips = get_num_changes_all_paths(test_test_same_1, changes_dict)
    test_test_diff_class_flips = get_num_changes_all_paths(test_test_diff, changes_dict)

    train_test_same_class_flips = get_num_changes_all_paths(train_test_same, changes_dict)
    train_test_same_class_0_flips = get_num_changes_all_paths(train_test_same_0, changes_dict)
    train_test_same_class_1_flips = get_num_changes_all_paths(train_test_same_1, changes_dict)
    train_test_diff_class_flips = get_num_changes_all_paths(train_test_diff, changes_dict)

    # ------------------------- calculate statistics ---------------------------------------

    # todo: include same vs diff without train/test filtering.
    #  code for that is still in same_diff_analysis file.

    stats_train_test_diff_same = {
        'same': {
            'train_train': {
                'num_paths': len(train_train_same_class_flips),
                'mean': np.mean(train_train_same_class_flips),
                'std': np.std(train_train_same_class_flips)
            },
            'test_test': {
                'num_paths': len(test_test_same_class_flips),
                'mean': np.mean(test_test_same_class_flips),
                'std': np.std(test_test_same_class_flips)
            },
            'train_test': {
                'num_paths': len(train_test_same_class_flips),
                'mean': np.mean(train_test_same_class_flips),
                'std': np.std(train_test_same_class_flips)
            }
        },
        'same_0': {
            'train_train': {
                'num_paths': len(train_train_same_class_0_flips),
                'mean': np.mean(train_train_same_class_0_flips),
                'std': np.std(train_train_same_class_0_flips)
            },
            'test_test': {
                'num_paths': len(test_test_same_class_0_flips),
                'mean': np.mean(test_test_same_class_0_flips),
                'std': np.std(test_test_same_class_0_flips)
            },
            'train_test': {
                'num_paths': len(train_test_same_class_0_flips),
                'mean': np.mean(train_test_same_class_0_flips),
                'std': np.std(train_test_same_class_0_flips)
            }
        },
        'same_1': {
            'train_train': {
                'num_paths': len(train_train_same_class_1_flips),
                'mean': np.mean(train_train_same_class_1_flips),
                'std': np.std(train_train_same_class_1_flips)
            },
            'test_test': {
                'num_paths': len(test_test_same_class_1_flips),
                'mean': np.mean(test_test_same_class_1_flips),
                'std': np.std(test_test_same_class_1_flips)
            },
            'train_test': {
                'num_paths': len(train_test_same_class_1_flips),
                'mean': np.mean(train_test_same_class_1_flips),
                'std': np.std(train_test_same_class_1_flips)
            },
        },
        'diff': {
            'train_train': {
                'num_paths': len(train_train_diff_class_flips),
                'mean': np.mean(train_train_diff_class_flips),
                'std': np.std(train_train_diff_class_flips)
            },
            'test_test': {
                'num_paths': len(test_test_diff_class_flips),
                'mean': np.mean(test_test_diff_class_flips),
                'std': np.std(test_test_diff_class_flips)
            },
            'train_test': {
                'num_paths': len(train_test_diff_class_flips),
                'mean': np.mean(train_test_diff_class_flips),
                'std': np.std(train_test_diff_class_flips)
            }
        }
    }
    