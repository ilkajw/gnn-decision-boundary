import os
import json
import numpy as np

from analyse_utils import get_num_changes_all_paths
from config import DATASET_NAME, CORRECTLY_CLASSIFIED_ONLY
from index_sets_utils import build_index_set_cuts


def stats_from_counts(counts):
    if not counts:
        return {"num_paths": 0, "mean": 0.0, "std": 0.0}
    return {
        "num_paths": len(counts),
        "mean": float(np.mean(counts)),
        "std": float(np.std(counts)),
    }


if __name__ == "__main__":

    # inputs
    split_path = "model/best_split.json"
    flips_path = f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_changes_per_path.json"
    out_path = f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_flip_stats_all_cuts.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # load precomputed flip history per path
    with open(flips_path, "r") as f:
        changes_dict = json.load(f)

    # build all pair-set cuts
    cuts = build_index_set_cuts(
        dataset_name=DATASET_NAME,
        correctly_classified_only=CORRECTLY_CLASSIFIED_ONLY,
        split_path=split_path,
    )

    # keys for index sets
    cut_keys = [
        # global by label
        "same_class_all", "same_class_0_all", "same_class_1_all", "diff_class_all",
        # train–train
        "train_train_same", "train_train_same_0", "train_train_same_1", "train_train_diff",
        # test–test
        "test_test_same",  "test_test_same_0",  "test_test_same_1",  "test_test_diff",
        # train–test
        "train_test_same", "train_test_same_0", "train_test_same_1", "train_test_diff",
    ]

    results = {}
    for key in cut_keys:
        pair_set = cuts[key]
        counts = get_num_changes_all_paths(pair_set, changes_dict)  # list: number of flips per path
        results[key] = stats_from_counts(counts)

    # save + print summary
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved flip statistics for {len(cut_keys)} cuts → {out_path}")