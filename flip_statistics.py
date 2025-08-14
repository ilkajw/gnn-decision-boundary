import os
import json
import numpy as np

from analyse_utils import get_num_changes_all_paths
from config import DATASET_NAME, CORRECTLY_CLASSIFIED_ONLY, DISTANCE_MODE
from index_sets_utils import build_index_set_cuts


def stats_from_counts(counts):
    if not counts:
        return {"num_paths": 0, "mean": 0.0, "median": 0.0, "std": 0.0, "min": 0, "max": 0}
    return {
        "num_paths": len(counts),
        "mean": float(np.mean(counts)),
        "median": float(np.median(counts)),
        "std": float(np.std(counts)),
        "min": float(np.min(counts)),
        "max": float(np.max(counts))
    }


if __name__ == "__main__":

    # define output path
    out_path = f"data/{DATASET_NAME}/analysis/flip_statistics/by_{DISTANCE_MODE}/" \
               f"{DATASET_NAME}_flip_stats_by_{DISTANCE_MODE}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # define inputs
    split_path = "model/best_split.json"

    # retrieve flip info per path according to distance mode
    if DISTANCE_MODE == "cost":
        flips_path = f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_flip_occurrences_per_path_by_cost.json"
    else:
        flips_path = f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_flip_occurrences_per_path_by_edit_step.json"

    # load precomputed flip history per path
    with open(flips_path, "r") as f:
        flips_dict = json.load(f)

    # build all pair-set cuts
    cuts = build_index_set_cuts(
        dataset_name=DATASET_NAME,
        correctly_classified_only=CORRECTLY_CLASSIFIED_ONLY,
        split_path=split_path,
    )

    # keys for index sets
    keys = [
        "same_class_all", "same_class_0_all", "same_class_1_all", "diff_class_all",
        "same_train_train", "same_0_train_train", "same_1_train_train", "diff_train_train",
        "same_test_test", "same_0_test_test", "same_1_test_test", "diff_test_test",
        "same_train_test", "same_0_train_test", "same_1_train_test", "diff_train_test",
    ]

    results = {}
    for key in keys:
        pair_set = cuts[key]
        counts = get_num_changes_all_paths(pair_set, flips_dict)  # list: number of flips per path
        results[key] = stats_from_counts(counts)
    # todo: add meta data to data
    # save + print summary
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved flip statistics for {len(keys)} cuts → {out_path}")