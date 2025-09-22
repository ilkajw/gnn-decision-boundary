# TODO: file descriptor

from datetime import datetime, timezone
from collections import defaultdict, Counter
import os
import json


from config import DATASET_NAME, ANALYSIS_DIR, MODEL_DIR, MODEL, CORRECTLY_CLASSIFIED_ONLY, DISTANCES_PATH, FLIPS_PATH,\
    DISTANCE_MODE
from index_sets_utils import build_index_set_cuts, graphs_correctly_classified

# ---- Set input, output parameters -----
split_path = os.path.join(MODEL_DIR, f"{DATASET_NAME}_{MODEL}_best_split.json")
output_dir = ANALYSIS_DIR
output_fname = f"{DATASET_NAME}_{MODEL}_flip_distribution_per_num_flips_by_{DISTANCE_MODE}.json"
max_num_flips = 10


# ----- Helpers -----
def flip_distribution_over_deciles_by_num_flips(
    max_num_flips,
    dist_input_path,
    flips_input_path,
    idx_pair_set=None,
    output_path=None,
    include_paths=False,
    correctly_classified_only=CORRECTLY_CLASSIFIED_ONLY,
):
    # TODO: docstring
    # To filter if defined
    if correctly_classified_only:
        correct = graphs_correctly_classified()

    # Load path distances and flips data
    with open(dist_input_path, "r") as f:
        distances = json.load(f)
    with open(flips_input_path, "r") as f:
        flips_per_path = json.load(f)

    def get_distance(i, j):
        s1, s2 = f"{i},{j}", f"{j},{i}"
        return distances.get(s1, distances.get(s2))

    # Accumulators
    abs_counts_by_k = {k: [0] * 10 for k in range(1, max_num_flips + 1)}
    flip_order_counts_by_k = {
        k: [[0] * 10 for _ in range(k)] for k in range(1, max_num_flips + 1)
    }
    ops_counts_by_k = {
        k: [Counter() for _ in range(10)] for k in range(1, max_num_flips + 1)
    }

    # To collect contributing paths
    paths_by_k = {k: [] for k in range(1, max_num_flips + 1)} if include_paths else None

    # To track number of contributing paths per k
    num_paths_by_k = defaultdict(int)

    for pair_str, flips in flips_per_path.items():

        # Skip paths without flips
        if not flips:
            continue
        i, j = map(int, pair_str.split(","))

        # Filter for index set given as argument
        if idx_pair_set is not None and (i, j) not in idx_pair_set and (j, i) not in idx_pair_set:
            continue

        # Filter for correctness. If index_set given, these have already been filtered
        if idx_pair_set is None and correctly_classified_only:
            if i not in correct or j not in correct:
                continue

        # Only consider num flips up to k
        k = len(flips)
        if k < 1 or k > max_num_flips:
            continue

        # Get path distance measure of (i, j) according to distance mode
        dist = get_distance(i, j)
        if dist is None:
            raise ValueError(f"[ERR] missing distance for ({i}, {j})")
        if not isinstance(dist, (int, float)):
            raise TypeError(f"[ERR] non-numeric distance for ({i}, {j}): {type(dist)}")
        if dist <= 0:
            raise ValueError(f"[ERR] non-positive distance for ({i}, {j}): {dist}")

        # Calculate per-path decile counts, split by flip order
        decile_counts = [0] * 10
        for flip_idx, flip in enumerate(flips):

            if not isinstance(flip, (list, tuple)) or len(flip) < 3:
                raise ValueError(
                    f"Flip entry must be a triple (step/cost, class, operation). Got: {flip} for pair {i},{j}")

            step, _lbl, op = flip

            if not isinstance(step, (int, float)) or step < 0:
                raise TypeError(f"'step/cost' must be non-negative number; got {step} (type={type(step)}) for {i},{j}")

            if not isinstance(op, str):
                raise TypeError(f"'operation' must be a string label, got {type(op)} for pair {i},{j}")

            rel = step / dist
            d = int(min(rel * 10, 9))  # bin 0...9, clamp 1.0 to 9
            decile_counts[d] += 1  # Accumulate absolutes
            flip_order_counts_by_k[k][flip_idx][d] += 1  # Accumulate per flip order counts
            ops_counts_by_k[k][d][op] += 1  # Accumulate operation counts by decile

        total_flips = sum(decile_counts)
        if total_flips == 0:
            continue

        # Accumulate absolutes
        acc = abs_counts_by_k[k]
        for d in range(10):
            acc[d] += decile_counts[d]

        num_paths_by_k[k] += 1
        if include_paths:
            paths_by_k[k].append({"pair": [i, j], "flips": flips})

    # Build result
    result = {}
    for k in range(1, max_num_flips + 1):
        abs_counts = abs_counts_by_k[k]
        total_abs = sum(abs_counts)
        global_prop = [c / total_abs if total_abs > 0 else 0.0 for c in abs_counts]

        # Per-flip-order distribution
        flip_order_distribution = {
            str(order + 1): {str(d): int(flip_order_counts_by_k[k][order][d]) for d in range(10)}
            for order in range(k)
        }

        # Operation distribution
        ops_by_decile = {}
        for d in range(10):
            decile_total = abs_counts[d]  # total flips in this decile for this k
            counter = ops_counts_by_k[k][d]
            abs_ops = {op: int(cnt) for op, cnt in counter.items()} if decile_total > 0 else {}
            norm_ops = {op: (cnt / decile_total) for op, cnt in counter.items()} if decile_total > 0 else {}
            ops_by_decile[str(d)] = {
                "abs": abs_ops,
                "norm": norm_ops,
            }

        # Store relative and absolute values per-k
        entry = {
            "num_paths": int(num_paths_by_k[k]),
            "abs": {str(d): int(abs_counts[d]) for d in range(10)},
            "norm": {str(d): float(global_prop[d]) for d in range(10)},
            "flip_order_distribution": flip_order_distribution,
            "ops_by_decile": ops_by_decile,
        }

        # Optionally store set of contributing paths
        if include_paths:
            entry["paths"] = paths_by_k[k]
        result[str(k)] = entry

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

    return result


# ---- Run analysis ----
if __name__ == "__main__":

    # Fail fast if inputs missing
    for p in [split_path, DISTANCES_PATH, FLIPS_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing input: {p}")

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

    # --- Per index set calculation ---

    per_index_set = {}
    for key in keys:
        idx_set = cuts[key]
        print(f"→ Computing per-num-flips decile distribution for {key} ({len(idx_set)} pairs)")
        stats = flip_distribution_over_deciles_by_num_flips(
            max_num_flips=max_num_flips,
            dist_input_path=DISTANCES_PATH,
            flips_input_path=FLIPS_PATH,
            output_path=None,
            idx_pair_set=idx_set,
            correctly_classified_only=CORRECTLY_CLASSIFIED_ONLY,
        )
        per_index_set[key] = {
            "num_pairs": len(idx_set),
            **stats
        }

    # --- Save to file ---

    data = {
        "meta": {
            "dataset": DATASET_NAME,
            "distance_mode": DISTANCE_MODE,
            "correctly_classified_only": CORRECTLY_CLASSIFIED_ONLY,
            "split_path": split_path,
            "dist_path": DISTANCES_PATH,
            "flips_path": FLIPS_PATH,
            "max_num_flips": max_num_flips,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "per_index_set": per_index_set
    }

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_fname)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved combined per-num-flips decile distributions → {output_path}")
