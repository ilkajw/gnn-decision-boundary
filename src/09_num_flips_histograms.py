"""
Build flips-per-path histograms for multiple index-set cuts.

Reads per-path flip lists (FLIPS_PATH) and index-set cuts, counts how many
paths have 0,1,2,... flips for each cut (absolute and relative frequencies),
and writes a consolidated JSON to ANALYSIS_DIR with per-index-set histograms.
Also saves small test JSONs listing paths with odd/even parity where relevant.

Requires config variables: DATASET_NAME, MODEL, MODEL_DIR, ANALYSIS_DIR,
CORRECTLY_CLASSIFIED_ONLY, DISTANCE_MODE, FLIPS_PATH.
Outputs: ANALYSIS_DIR/<DATASET_NAME>_<MODEL>_flips_hist_by_<DISTANCE_MODE>.json
(and test JSONs in ANALYSIS_DIR/test).
"""

import os
import json
from collections import defaultdict

from datetime import datetime, timezone
from config import DATASET_NAME, MODEL, MODEL_DIR, ANALYSIS_DIR, CORRECTLY_CLASSIFIED_ONLY, DISTANCE_MODE, FLIPS_PATH
from index_sets_utils import build_index_set_cuts


# ---- Set input, output params ----
split_path = os.path.join(MODEL_DIR, f"{DATASET_NAME}_{MODEL}_best_split.json")

output_dir = ANALYSIS_DIR
output_fname = f"{DATASET_NAME}_{MODEL}_flips_hist_by_{DISTANCE_MODE}.json"

# To save lists of paths incorrectly having even/odd number of flips
test_output_dir = os.path.join(ANALYSIS_DIR, "test")


# ---- Helper functions ----
def to_relative(counts_dict):
    # counts_dict: {"0": int, "1": int, ...} or {0: int, 1: int, ...}
    total = sum(counts_dict.values())
    if total == 0:
        return {str(k): 0.0 for k in counts_dict.keys()}
    return {str(k): (counts_dict[k] / total) for k in counts_dict.keys()}


def count_paths_by_num_flips(
        idx_pair_set,
        flips_input_path,
        output_path=None,
        same_class=False
):
    """
    For a given set of index pairs, count how many paths have 0, 1, 2, ... flips.

    Args:
        idx_pair_set (set of tuples): Set of (i, j) graph index pairs to consider.
        flips_input_path (str): Path to JSON file with flip data like {"i,j": [[step, label], ...], ...}
        output_path (str, optional): Where to save the resulting histogram (JSON). If None, don't save.
        same_class (boolean): If the given idx_pair_set is of same_class or diff_class category.

    Returns:
        dict: {num_flips: count} showing how many paths have that many flips.
    """
    # For testing only
    same_class_odd_flips = []
    diff_class_even_flips = []

    # Load flip data from file
    with open(flips_input_path) as f:
        flips_per_path = json.load(f)

    # Initialize histogram
    flip_histogram = defaultdict(int)

    # Count paths per number of flips
    for pair_str, flips in flips_per_path.items():
        i, j = map(int, pair_str.split(","))

        if (i, j) not in idx_pair_set and (j, i) not in idx_pair_set:
            continue

        num_flips = len(flips)

        # For testing only
        if same_class and num_flips % 2 == 1:
            same_class_odd_flips.append((i, j))
        if not same_class and num_flips % 2 == 0:
            diff_class_even_flips.append((i, j))

        flip_histogram[num_flips] += 1

    # Optionally save
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(dict(flip_histogram), f, indent=2)

    os.makedirs(test_output_dir, exist_ok=True)

    if same_class:
        output_path = os.path.join(test_output_dir, f"{DATASET_NAME}_{MODEL}_same_class_odd_flips.json")
        with open(output_path, "w") as f:
            json.dump(same_class_odd_flips, f, indent=2)
    if not same_class:
        output_path = os.path.join(test_output_dir, f"{DATASET_NAME}_{MODEL}_diff_class_even_flips.json")
        with open(output_path, "w") as f:
            json.dump(diff_class_even_flips, f, indent=2)

    return dict(flip_histogram)


if __name__ == "__main__":

    # Fail fast if inputs missing
    for p in [split_path, FLIPS_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing input: {p}")

    # Build all index-set cuts (same/diff + train/train, test/test, train/test)
    idx_pair_sets = build_index_set_cuts(
        dataset_name=DATASET_NAME,
        correctly_classified_only=CORRECTLY_CLASSIFIED_ONLY,
        split_path=split_path,
    )

    # key -> whether this cut is "same_class" (affects parity checks inside count_paths_by_num_flips)
    keys_and_flags = [
        # global by label
        ("same_class_all",   True),
        ("same_class_0_all", True),
        ("same_class_1_all", True),
        ("diff_class_all",   False),

        # train–train
        ("same_train_train",   True),
        ("same_0_train_train", True),
        ("same_1_train_train", True),
        ("diff_train_train",   False),

        # test–test
        ("same_test_test",   True),
        ("same_0_test_test", True),
        ("same_1_test_test", True),
        ("diff_test_test",   False),

        # train–test
        ("same_train_test",   True),
        ("same_0_train_test", True),
        ("same_1_train_test", True),
        ("diff_train_test",   False),
    ]

    data = {
        "meta": {
            "dataset": DATASET_NAME,
            "model": MODEL,
            "distance_mode": DISTANCE_MODE,
            "correctly_classified_only": CORRECTLY_CLASSIFIED_ONLY,
            "split_path": split_path,
            "flips_path": FLIPS_PATH,
            "generated_at": datetime.now(timezone.utc).isoformat()
        },
        "results": {}
    }

    # Run histograms for every index set
    for key, same_flag in keys_and_flags:

        # Retrieve index pairs for index set
        idx_pair_set = idx_pair_sets[key]
        print(f"→ counting flips histogram for {key} ({len(idx_pair_set)} pairs)")

        # Calculate histogram with absolute values
        hist_abs = count_paths_by_num_flips(
            idx_pair_set=idx_pair_set,
            flips_input_path=FLIPS_PATH,
            output_path=None,
            same_class=same_flag,
        )

        # Calculate relative values
        hist_rel = to_relative(hist_abs)

        # Records for this index set
        data["results"][key] = {
            "num_pairs": len(idx_pair_set),
            "hist_abs": hist_abs,  # {num_flips: count}
            "hist_rel": hist_rel,  # {num_flips: proportion}
        }

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_fname)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved consolidated results → {output_path}")
