import json
import numpy as np

from index_sets import *


def get_change_counts(pairs):
    counts = []
    for i, j in pairs:
        key = f"{i},{j}"
        if key in changes_dict:
            counts.append(len(changes_dict[key]))
        elif f"{j},{i}" in changes_dict:  # in case direction was flipped
            counts.append(len(changes_dict[f"{j},{i}"]))
    return counts


if __name__ == "__main__":

    dataset_name = "MUTAG"

    with open(f"data/{dataset_name}/predictions/{dataset_name}_edit_path_predictions.json", "r") as f:
        predictions = json.load(f)

    with open(f"data/{dataset_name}/predictions/{dataset_name}_changes_per_path.json") as f:
        changes_dict = json.load(f)

    same_class_pairs = graph_index_pairs_diff_class(dataset_name=dataset_name,
                                                    correctly_classified_only=True,
                                                    save_path=f"data/{dataset_name}/index_sets/{dataset_name}_idx_pairs_diff_class.json")

    diff_class_pairs = graph_index_pairs_same_class(dataset_name=dataset_name,
                                                    correctly_classified_only=True,
                                                    save_path=f"data/{dataset_name}/index_sets/{dataset_name}_idx_pairs_same_class.json")

    same_changes = get_change_counts(same_class_pairs)
    diff_changes = get_change_counts(diff_class_pairs)

    stats_num_changes = {
        "same_class": {
            "num_pairs": len(same_changes),
            "mean": np.mean(same_changes) if same_changes else 0,
            "std": np.std(same_changes) if same_changes else 0
        },
        "diff_class": {
            "count": len(diff_changes),
            "mean": np.mean(diff_changes) if diff_changes else 0,
            "std": np.std(diff_changes) if diff_changes else 0
        }
    }

    print(json.dumps(stats_num_changes, indent=2))

    # todo: save, test
