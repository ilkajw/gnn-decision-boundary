import json
import numpy as np
from config import DATASET_NAME
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

    save_path = f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_changes_per_path_same_vs_diff_class.json"

    with open(f"data/{DATASET_NAME}/predictions/{DATASET_NAME}_edit_path_predictions.json", "r") as f:
        predictions = json.load(f)

    with open(f"data/{DATASET_NAME}/predictions/{DATASET_NAME}_changes_per_path.json") as f:
        changes_dict = json.load(f)

    # get all index pairs of same and different classes
    same_class_pairs = graph_index_pairs_diff_class(dataset_name=DATASET_NAME,
                                                    correctly_classified_only=True,
                                                    save_path=f"data/{DATASET_NAME}/index_sets/{DATASET_NAME}_idx_pairs_diff_class.json")

    diff_class_pairs = graph_index_pairs_same_class(dataset_name=DATASET_NAME,
                                                    correctly_classified_only=True,
                                                    save_path=f"data/{DATASET_NAME}/index_sets/{DATASET_NAME}_idx_pairs_same_class.json")

    # get lists of change count per path graph sequence for source and target
    # belonging to same or different classes
    same_changes = get_change_counts(same_class_pairs)
    diff_changes = get_change_counts(diff_class_pairs)

    # calculate statistics
    stats_changes_per_path = {
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

    # save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(stats_changes_per_path, f, indent=2)
