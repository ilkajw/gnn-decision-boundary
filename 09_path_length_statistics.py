# path_length_statistics.py

import json
import os
import numpy as np

from config import (
    DATASET_NAME, DISTANCE_MODE, MODEL, MODEL_DIR, ANALYSIS_DIR,
    CORRECTLY_CLASSIFIED_ONLY
)
from index_sets_utils import build_index_set_cuts, graphs_correctly_classified


# set if to calculate for all paths or paths between correctly classified endpoints only
correctly_classified_only = False

# ---- set input, output paths ----

split_path = f"{MODEL_DIR}/{DATASET_NAME}_{MODEL}_best_split.json"
output_dir = ANALYSIS_DIR
output_fname = f"{DATASET_NAME}_{MODEL}_path_length_stats_by_{DISTANCE_MODE}.json"


# ----- helpers ----
def load_lengths_from_precalc(
        dist_path: str,
        idx_set=None,
        correctly_classified_only: bool = CORRECTLY_CLASSIFIED_ONLY
        ):
    """
    Load path 'lengths' (cost or #ops) from a single JSON:
      {"i,j": value, ...}
    and optionally filter to an index-pair set and correctly-classified graphs.
    Returns: {(i,j): value, ...}
    """
    with open(dist_path, "r") as f:
        raw = json.load(f)

    # optional filter for correctness (global case only)
    correct = set()
    if correctly_classified_only and idx_set is None:
        correct = set(graphs_correctly_classified())

    out = {}
    for pair_str, val in raw.items():
        i, j = map(int, pair_str.split(","))
        # filter by idx_set (unordered)
        if idx_set is not None and (i, j) not in idx_set and (j, i) not in idx_set:
            continue
        # global correctness filter (per-index-set cuts already did this upstream)
        if correctly_classified_only and idx_set is None:
            if i not in correct or j not in correct:
                continue
        out[(i, j)] = float(val)
    return out


def path_length_statistics(path_lengths: dict):
    vals = list(path_lengths.values())
    if not vals:
        return {"num_paths": 0, "mean": 0.0, "std_dev": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "num_paths": len(vals),
        "mean":   float(np.mean(vals)),
        "std_dev":float(np.std(vals)),
        "median": float(np.median(vals)),
        "min":    float(np.min(vals)),
        "max":    float(np.max(vals)),
    }


# ---- run ----
if __name__ == "__main__":

    # pick the right precalc file
    if DISTANCE_MODE == "cost":
        dist_path = os.path.join("data_actual_best\MUTAG\GAT/analysis/by_cost", f"{DATASET_NAME}_dist_per_path.json")
    elif DISTANCE_MODE == "edit_step":
        dist_path = os.path.join("data_actual_best\MUTAG\GAT/analysis/by_cost", f"{DATASET_NAME}_num_ops_per_path.json")
    else:
        print(f"[warn] unexpected DISTANCE_MODE='{DISTANCE_MODE}', defaulting to 'cost'")
        dist_path = os.path.join(ANALYSIS_DIR, f"{DATASET_NAME}_dist_per_path.json")

    # build all index-set cuts
    idx_pair_sets = build_index_set_cuts(
        dataset_name=DATASET_NAME,
        correctly_classified_only=correctly_classified_only,
        split_path=split_path,
    )

    # ---- global stats ----
    all_lengths = load_lengths_from_precalc(
        dist_path=dist_path,
        idx_set=None,
        correctly_classified_only=correctly_classified_only
    )
    all_stats = path_length_statistics(all_lengths)

    # ---- per-index-set stats ----
    per_set_stats = {}
    for key, idx_set in idx_pair_sets.items():
        lengths = load_lengths_from_precalc(
            dist_path=dist_path,
            idx_set=idx_set,
            correctly_classified_only=correctly_classified_only
        )
        per_set_stats[key] = {
            **path_length_statistics(lengths),
            "num_pairs_total": int(len(idx_set)),
            "num_pairs_with_data": int(len(lengths)),
            "coverage": float(len(lengths) / max(1, len(idx_set))),
        }

    # save
    data = {
        "dataset": DATASET_NAME,
        "distance_mode": DISTANCE_MODE,
        "source_file": dist_path,
        "global": all_stats,
        "per_index_set": per_set_stats,
    }
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_fname)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved combined stats → {out_path}")
