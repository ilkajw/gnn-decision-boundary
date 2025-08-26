import os
import json
from config import DATASET_NAME, DISTANCE_MODE
from index_sets_utils import build_index_set_cuts


# ------- define input, output paths ---------

dist_path = f"data_control/{DATASET_NAME}/analysis/{DATASET_NAME}_{'dist_per_path.json' if DISTANCE_MODE == 'cost' else 'num_ops_per_path.json'}"
flips_path = f"data_control/{DATASET_NAME}/analysis/{DATASET_NAME}_flip_occurrences_per_path_by_{DISTANCE_MODE}.json"

output_dir = f"data_control/{DATASET_NAME}/analysis/flip_distributions_k2"
output_fname = f"{DATASET_NAME}_first_second_flips_by_{DISTANCE_MODE}.json"


# ------------ helpers -----------------

def load_distances():
    with open(dist_path) as f:
        return json.load(f)


def load_flips():
    with open(flips_path) as f:
        return json.load(f)


def get_distance(i, j, distances):
    s1, s2 = f"{i},{j}", f"{j},{i}"
    return distances.get(s1, distances.get(s2))


def to_decile(step, dist):
    if dist is None or dist == 0:
        return None
    rel = step / dist
    return int(min(rel * 10, 9))  # clamp at 9


def first_second_flip_distribution(idx_pair_set, distances, flips_per_path):
    """Compute distribution of first and second flips over deciles."""
    first_counts = [0] * 10
    second_counts = [0] * 10
    num_paths = 0

    for pair_str, flips in flips_per_path.items():

        # filter for paths with 2 flips only
        if len(flips) != 2:
            continue

        # extract indices from key
        i, j = map(int, pair_str.split(","))

        # filter for given index pair set
        if (i, j) not in idx_pair_set and (j, i) not in idx_pair_set:
            continue

        dist = get_distance(i, j, distances)
        if not dist:
            print(f"[warn] no distance found for ({i}, {j})")
            continue

        # calculate tenth of the path flips fall in
        first_dec = to_decile(flips[0][0], dist)
        second_dec = to_decile(flips[1][0], dist)
        if first_dec is None:
            print(f"[warn] couldn't calculate relative position of first flip for ({i}, {j}).")
            continue
        if second_dec is None:
            print(f"[warn] couldn't calculate relative position of second flip for ({i}, {j}).")
            continue

        first_counts[first_dec] += 1
        second_counts[second_dec] += 1
        num_paths += 1

    def normalize(counts):
        total = sum(counts)
        return [c / total if total else 0.0 for c in counts]

    return {
        "num_paths": num_paths,
        "first": {
            "abs_counts": {str(d): first_counts[d] for d in range(10)},
            "proportion": {str(d): normalize(first_counts)[d] for d in range(10)},
        },
        "second": {
            "abs_counts": {str(d): second_counts[d] for d in range(10)},
            "proportion": {str(d): normalize(second_counts)[d] for d in range(10)},
        },
    }


# ---------- run analysis -------------

if __name__ == "__main__":

    # load input data
    distances = load_distances()
    flips_per_path = load_flips()
    cuts = build_index_set_cuts()

    # todo: extract relevant groups here and delete extraction from plotting
    # calculate distribution per index set
    results = {}
    for group_name, idx_pairs in cuts.items():
        res = first_second_flip_distribution(idx_pairs, distances, flips_per_path)
        results[group_name] = res
        print(f"[{group_name}] {res['num_paths']} paths with 2 flips")

    # save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_fname)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {out_path}")
