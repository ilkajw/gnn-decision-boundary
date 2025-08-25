import json
import os
import re
import torch
import numpy as np

from config import DATASET_NAME, DISTANCE_MODE, CORRECTLY_CLASSIFIED_ONLY
from index_sets_utils import build_index_set_cuts

# --------- define input, output params ------------

graph_sequence_dir = f"data_control/{DATASET_NAME}/predictions/edit_path_graphs_with_predictions_CUMULATIVE_COST"
split_path = "model_control/best_split.json"

output_dir = f"data_control/{DATASET_NAME}/analysis/path_lengths/"
output_fname = f"{DATASET_NAME}_path_length_stats_by_{DISTANCE_MODE}.json"


# --------- helpers --------------

def path_lengths(seq_dir, out_path=None, idx_set=None):

    pattern = re.compile(r"g(\d+)_to_g(\d+)_it\d+_graph_sequence\.pt")
    length_dict = {}

    for fname in os.listdir(seq_dir):

        if not fname.endswith(".pt"):
            continue

        match = pattern.match(fname)
        if not match:
            continue

        i, j = int(match.group(1)), int(match.group(2))

        if idx_set:
            if (i, j) not in idx_set and (j, i) not in idx_set:
                continue

        filepath = os.path.join(seq_dir, fname)

        sequence = torch.load(filepath, weights_only=False)

        if DISTANCE_MODE == "cost":
            val = getattr(sequence[-1], "cumulative_cost", None)
            length_dict[(i, j)] = val
        else:
            length_dict[(i, j)] = len(sequence)

    # make dict serialization for json dump
    serializable = {f"{i},{j}": v for (i, j), v in length_dict.items()}

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(serializable, f, indent=2)

    return serializable


def path_length_statistics(path_lengths, out_path=None):

    vals = list(path_lengths.values())

    stats = {"num_paths": len(vals),
             "mean": np.mean(vals),
             "std_dev": np.std(vals),
             "median": np.median(vals),
             "min": np.min(vals),
             "max": np.max(vals)
             }

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(stats, f, indent=2)

    return stats


# -------- run analysis ----------

if __name__ == "__main__":

    idx_pair_set = build_index_set_cuts(
        dataset_name=DATASET_NAME,
        correctly_classified_only=CORRECTLY_CLASSIFIED_ONLY,  # todo: evaluate for others too
        split_path=split_path,
    )

    # --------- take global stats -----------

    all_path_lens = path_lengths(seq_dir=graph_sequence_dir, out_path=None)
    all_stats = path_length_statistics(path_lengths=all_path_lens, out_path=None)

    # -------- take per-index set stats ---------

    per_set_stats = {}

    for key in idx_pair_set.keys():
        idx_set = idx_pair_set[key]
        path_lens = path_lengths(seq_dir=graph_sequence_dir, idx_set=idx_pair_set[key])
        per_set_stats[key] = path_length_statistics(path_lengths=path_lens)

    data = {
        "dataset": DATASET_NAME,
        "distance_mode": DISTANCE_MODE,
        "global": all_stats,
        "per_index_set": per_set_stats,
    }

    # save
    combined_path = os.path.join(
        output_dir,
        output_fname
    )

    os.makedirs(output_dir, exist_ok=True)
    with open(combined_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved combined stats to {combined_path}")
