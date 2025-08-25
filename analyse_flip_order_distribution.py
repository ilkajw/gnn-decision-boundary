from datetime import datetime, timezone
from collections import defaultdict
import os
import json

from config import DATASET_NAME, CORRECTLY_CLASSIFIED_ONLY, DISTANCE_MODE
from index_sets_utils import build_index_set_cuts, graphs_correctly_classified

# -------- define input, output params ---------

split_path = "model_control/best_split.json"
output_dir = f"data_control/{DATASET_NAME}/analysis/paths_per_num_flips/by_{DISTANCE_MODE}/"
output_fname = f"{DATASET_NAME}_flip_distribution_per_num_flips_by_{DISTANCE_MODE}.json"
max_num_flips = 10

# retrieve data according to distance mode set in config
if DISTANCE_MODE == "cost":
    dist_path = f"data_control/{DATASET_NAME}/analysis/{DATASET_NAME}_dist_per_path.json"
    flips_path = f"data_control/{DATASET_NAME}/analysis/{DATASET_NAME}_flip_occurrences_per_path_by_cost.json"

elif DISTANCE_MODE == "num_ops":
    dist_path = f"data_control/{DATASET_NAME}/analysis/{DATASET_NAME}_num_ops_per_path.json"
    flips_path = f"data_control/{DATASET_NAME}/analysis/{DATASET_NAME}_flip_occurrences_per_path_by_edit_step.json"

else:
    print(f"[warn] config.DISTANCE_MODE has unexpected value '{DISTANCE_MODE}'. Expected 'cost' or 'num_ops'."
          f"Assuming 'cost'.")
    dist_path = f"data_control/{DATASET_NAME}/analysis/{DATASET_NAME}_dist_per_path.json"
    flips_path = f"data_control/{DATASET_NAME}/analysis/{DATASET_NAME}_flip_occurrences_per_path_by_cost.json"

# ---------------- helpers --------------------

def flip_distribution_over_deciles_by_num_flips(
    max_num_flips,
    dist_input_path,
    flips_input_path,
    idx_pair_set=None,
    output_path=None,
    include_paths=False
):
    # to filter if defined
    if CORRECTLY_CLASSIFIED_ONLY:
        correct = graphs_correctly_classified()

    # load path distance and flips data
    with open(dist_input_path, "r") as f:
        distances = json.load(f)
    with open(flips_input_path, "r") as f:
        flips_per_path = json.load(f)

    def get_distance(i, j):
        s1, s2 = f"{i},{j}", f"{j},{i}"
        return distances.get(s1, distances.get(s2))

    # accumulators
    abs_counts_by_k = {k: [0] * 10 for k in range(1, max_num_flips + 1)}
    flip_order_counts_by_k = {
        k: [[0] * 10 for _ in range(k)] for k in range(1, max_num_flips + 1)
    }
    paths_by_k = {k: [] for k in range(1, max_num_flips + 1)} if include_paths else None
    num_paths_by_k = defaultdict(int)

    for pair_str, flips in flips_per_path.items():
        if not flips:
            continue
        i, j = map(int, pair_str.split(","))

        # filter for index set
        if idx_pair_set is not None and (i, j) not in idx_pair_set and (j, i) not in idx_pair_set:
            continue

        # filter for correctness. for index_sets, these have already been filtered
        if idx_pair_set is None and CORRECTLY_CLASSIFIED_ONLY:
            if i not in correct or j not in correct:
                continue

        # only consider num flips up to k
        k = len(flips)
        if k < 1 or k > max_num_flips:
            continue

        # get path distance of i, j according to distance mode
        dist = get_distance(i, j)
        if not dist:  # None or 0
            print(f"[WARN] missing distance for {i}, {j}")
            continue

        # calculate per-path decile counts, split by flip order
        decile_counts = [0] * 10
        for flip_idx, (step, _lbl) in enumerate(flips):
            rel = step / dist
            d = int(min(rel * 10, 9))  # bin 0..9, clamp 1.0 to 9
            decile_counts[d] += 1
            # per-flip-order
            flip_order_counts_by_k[k][flip_idx][d] += 1

        total_flips = sum(decile_counts)
        if total_flips == 0:
            continue

        # accumulate absolutes (all flips lumped)
        acc = abs_counts_by_k[k]
        for d in range(10):
            acc[d] += decile_counts[d]

        num_paths_by_k[k] += 1
        if include_paths:
            paths_by_k[k].append({"pair": [i, j], "flips": flips})

    # build result
    result = {}
    for k in range(1, max_num_flips + 1):
        abs_counts = abs_counts_by_k[k]
        total_abs = sum(abs_counts)
        if total_abs > 0:
            global_prop = [c / total_abs for c in abs_counts]
        else:
            global_prop = [0.0] * 10

        # per-flip-order distribution
        flip_order_distribution = {
            str(order + 1): {str(d): int(flip_order_counts_by_k[k][order][d]) for d in range(10)}
            for order in range(k)
        }

        # store relative and absolute values per-k
        entry = {
            "num_paths": int(num_paths_by_k[k]),
            "abs": {str(d): int(abs_counts[d]) for d in range(10)},
            "norm": {str(d): float(global_prop[d]) for d in range(10)},
            "flip_order_distribution": flip_order_distribution,
        }
        # optionally add set of contributing paths
        if include_paths:
            entry["paths"] = paths_by_k[k]
        result[str(k)] = entry

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

    return result


# -------- run analysis ---------------
if __name__ == "__main__":

    cuts = build_index_set_cuts(
        correctly_classified_only=CORRECTLY_CLASSIFIED_ONLY,
        split_path=split_path,
    )

    keys = [
        "same_class_all", "same_class_0_all", "same_class_1_all", "diff_class_all",
        "same_train_train", "same_0_train_train", "same_1_train_train", "diff_train_train",
        "same_test_test", "same_0_test_test", "same_1_test_test", "diff_test_test",
        "same_train_test", "same_0_train_test", "same_1_train_test", "diff_train_test",
    ]

    # ------------ global --------------
    print("→ Computing per-num-flips decile distribution (GLOBAL)")
    global_stats = flip_distribution_over_deciles_by_num_flips(
        max_num_flips=max_num_flips,
        dist_input_path=dist_path,
        flips_input_path=flips_path,
        output_path=None,
        idx_pair_set=None,
    )

    # ------------ per index set --------------
    per_index_set = {}
    for key in keys:
        idx_set = cuts[key]
        print(f"→ Computing per-num-flips decile distribution for {key} ({len(idx_set)} pairs)")
        stats = flip_distribution_over_deciles_by_num_flips(
            max_num_flips=max_num_flips,
            dist_input_path=dist_path,
            flips_input_path=flips_path,
            output_path=None,
            idx_pair_set=idx_set,
        )
        per_index_set[key] = {
            "num_pairs": len(idx_set),
            **stats
        }

    # ---------- save to file -------------
    data = {
        "meta": {
            "dataset": DATASET_NAME,
            "distance_mode": DISTANCE_MODE,
            "correctly_classified_only": CORRECTLY_CLASSIFIED_ONLY,
            "split_path": split_path,
            "dist_path": dist_path,
            "flips_path": flips_path,
            "max_num_flips": max_num_flips,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "global": global_stats,
        "per_index_set": per_index_set
    }

    # save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_fname)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved combined per-num-flips decile distributions → {output_path}")
