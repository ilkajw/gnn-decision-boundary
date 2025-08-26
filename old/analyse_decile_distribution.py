import json
import os
import numpy as np
from collections import defaultdict
from datetime import datetime, timezone

from config import DATASET_NAME, CORRECTLY_CLASSIFIED_ONLY, DISTANCE_MODE
from index_sets_utils import build_index_set_cuts, graphs_correctly_classified

# ------- define input, output paths ------------

split_path = "../model_control/best_split.json"

output_dir = f"data_control/{DATASET_NAME}/analysis/decile_distribution/by_{DISTANCE_MODE}"
output_fname = f"{DATASET_NAME}_flip_distribution_STATS_by_{DISTANCE_MODE}.json"


# ---------------- helpers ----------------

def flip_distribution_over_deciles_by_indexset(idx_pair_set, dist_input_path, flips_input_path, output_path=None):

    if CORRECTLY_CLASSIFIED_ONLY:
        correct = graphs_correctly_classified()

    # load
    with open(dist_input_path) as f:
        distances = json.load(f)
    with open(flips_input_path) as f:
        flips_per_path = json.load(f)

    # accumulators
    per_decile_per_path = defaultdict(list)  # d -> [proportions from each path]
    abs_counts = [0]*10                      # global flip totals per decile
    num_paths = 0

    def get_distance(i, j):
        s1, s2 = f"{i},{j}", f"{j},{i}"
        return distances.get(s1, distances.get(s2))

    for pair_str, flips in flips_per_path.items():
        if not flips:
            continue
        i, j = map(int, pair_str.split(","))

        if idx_pair_set is not None:
            if (i, j) not in idx_pair_set and (j, i) not in idx_pair_set:
                continue

        # filter for correctness. for index_sets, these have already been filtered
        if idx_pair_set is None:
            if CORRECTLY_CLASSIFIED_ONLY:
                if i not in correct or j not in correct:
                    continue

        dist = get_distance(i, j)
        if not dist:
            continue

        # per-path decile counts
        decile_counts = [0]*10
        for step, _ in flips:
            rel = step / dist
            d = int(min(rel * 10, 9))  # 0..9, clamp 1.0 to 9
            decile_counts[d] += 1

        total = sum(decile_counts)
        if total == 0:
            continue

        # per-path proportions with equal path weight
        for d in range(10):
            per_decile_per_path[d].append(decile_counts[d] / total)

        # global absolute accumulation (flip weight)
        for d in range(10):
            abs_counts[d] += decile_counts[d]

        num_paths += 1

    # average per-path
    avg_per_path = {
        str(d): (float(np.mean(per_decile_per_path[d])) if per_decile_per_path[d] else 0.0)
        for d in range(10)
    }

    # global flip-weighted distribution
    total_abs = sum(abs_counts)
    global_proportion = {
        str(d): (abs_counts[d] / total_abs if total_abs else 0.0) for d in range(10)
    }

    result = {
        "num_paths": num_paths,
        "avg_per_path": avg_per_path,
        "abs_counts": {str(d): int(abs_counts[d]) for d in range(10)},
        "global_proportion": global_proportion,
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

    return result


# --------- run analysis ----------

if __name__ == "__main__":

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_fname)

    # retrieve data according to distance mode (cost vs. num_ops)
    if DISTANCE_MODE == "cost":
        dist_path = f"data_control/{DATASET_NAME}/analysis/{DATASET_NAME}_dist_per_path.json"
        flips_path = f"data_control/{DATASET_NAME}/analysis/{DATASET_NAME}_flip_occurrences_per_path_by_cost.json"

    elif DISTANCE_MODE == "cost":
        dist_path = f"data_control/{DATASET_NAME}/analysis/{DATASET_NAME}_num_ops_per_path.json"
        flips_path = f"data_control/{DATASET_NAME}/analysis/{DATASET_NAME}_flip_occurrences_per_path_by_edit_step.json"

    else:
        print(f"[warn] config.DISTANCE_MODE has unexpected value {DISTANCE_MODE}. Expected 'cost' or 'num_ops."
              f"Assuming 'cost.")
        dist_path = f"data_control/{DATASET_NAME}/analysis/{DATASET_NAME}_dist_per_path.json"
        flips_path = f"data_control/{DATASET_NAME}/analysis/{DATASET_NAME}_flip_occurrences_per_path_by_cost.json"

    # build all cut index sets (same/diff class + train-train / test-test / train-test)
    cuts = build_index_set_cuts(
        dataset_name=DATASET_NAME,
        correctly_classified_only=CORRECTLY_CLASSIFIED_ONLY,
        split_path=split_path,
    )

    keys = [
        "same_class_all", "same_class_0_all", "same_class_1_all", "diff_class_all",
        "same_train_train", "same_0_train_train", "same_1_train_train", "diff_train_train",
        "same_test_test",  "same_0_test_test",  "same_1_test_test",  "diff_test_test",
        "same_train_test", "same_0_train_test", "same_1_train_test", "diff_train_test",
    ]

    # -------------- global distribution ----------------

    all_stats = flip_distribution_over_deciles_by_indexset(
        idx_pair_set=None,
        dist_input_path=dist_path,
        flips_input_path=flips_path
    )

    # ----------- per index set distribution -----------

    per_set_stats = {}
    for key in keys:
        idx_set = cuts[key]
        print(f"Computing decile distribution for {key} ({len(idx_set)} pairs)")
        per_set_stats[key] = flip_distribution_over_deciles_by_indexset(
            idx_pair_set=idx_set,
            dist_input_path=dist_path,
            flips_input_path=flips_path,
            output_path=None,
        )

    # combine info on all index sets
    combined = {
        "meta": {
            "dataset": DATASET_NAME,
            "distance_mode": DISTANCE_MODE,
            "correctly_classified_only": CORRECTLY_CLASSIFIED_ONLY,
            "split_path": split_path,
            "dist_path": dist_path,
            "flips_path": flips_path,
            "generated_at": datetime.now(timezone.utc).isoformat()
        },
        "global": all_stats,
        "per_index_set": per_set_stats,
    }

    # ------------ save ------------

    out_path = os.path.join(
        output_dir,
        f"{DATASET_NAME}_flip_distribution_STATS_by_{DISTANCE_MODE}.json"
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"Saved combined flip distribution stats to {out_path}")
