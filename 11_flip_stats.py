"""
Summarize per-path flip counts and produce per-index-set statistics.

Reads per-path flip lists from FLIPS_PATH, counts flips per path, and
aggregates simple statistics (count, mean, median, std, min, max)
for each index-set cut produced by index_sets_utils.build_index_set_cuts.
Writes a JSON summary to ANALYSIS_DIR/<DATASET_NAME>_<MODEL>_flip_stats_by_<DISTANCE_MODE>.json
containing a "meta" block and "per_index_set" results.

Requires config variables: DATASET_NAME, MODEL, MODEL_DIR, ANALYSIS_DIR,
CORRECTLY_CLASSIFIED_ONLY, DISTANCE_MODE, FLIPS_PATH.
"""

import os
import json
import numpy as np
from datetime import datetime, timezone

from config import DATASET_NAME, MODEL, MODEL_DIR, ANALYSIS_DIR, CORRECTLY_CLASSIFIED_ONLY, DISTANCE_MODE, FLIPS_PATH
from index_sets_utils import build_index_set_cuts


# ---- Define input, output paths -----
split_path = os.path.join(MODEL_DIR, f"{DATASET_NAME}_{MODEL}_best_split.json")
output_dir = f"{ANALYSIS_DIR}"
output_fname = f"{DATASET_NAME}_{MODEL}_flip_stats_by_{DISTANCE_MODE}.json"


# ----- Helpers -----
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

def get_num_flips_for_idxset_paths(pairs, changes_dict):
    counts = []
    for i, j in pairs:
        key = f"{i},{j}"
        if key in changes_dict:
            counts.append(len(changes_dict[key]))
        elif f"{j},{i}" in changes_dict:  # in case direction was flipped, should not happen though
            counts.append(len(changes_dict[f"{j},{i}"]))
        else:
            continue
    return counts


# ----- run ------

if __name__ == "__main__":

    # Fail fast if inputs missing
    for p in [split_path, FLIPS_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing input: {p}")

    # Load precomputed flip history per path
    with open(FLIPS_PATH, "r") as f:
        flips_dict = json.load(f)

    # Build all index-pair set cuts as 'cut_name' -> 'list of graph pairs included'
    index_sets = build_index_set_cuts(
        correctly_classified_only=CORRECTLY_CLASSIFIED_ONLY,
        split_path=split_path,
    )

    results = {}
    for key in index_sets.keys():
        pair_set = index_sets[key]
        counts = get_num_flips_for_idxset_paths(pairs=pair_set, changes_dict=flips_dict)  # list: number flips per path
        results[key] = {
            "num_pairs": int(len(pair_set)),
            "coverage": float(len(counts) / max(1, len(pair_set))),
            **stats_from_counts(counts),
        }

    # Summarize results
    data = {
        "meta": {
            "dataset": DATASET_NAME,
            "distance_mode": DISTANCE_MODE,
            "correctly_classified_only": CORRECTLY_CLASSIFIED_ONLY,
            "split_path": split_path,
            "flips_path": FLIPS_PATH,
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
        "per_index_set": results,
    }

    # Save summary
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_fname)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved flip statistics for {len(index_sets.keys())} cuts → {output_path}")
