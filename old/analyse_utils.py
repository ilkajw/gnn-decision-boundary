import sys
import os
# add submodule root to Python path
submodule_path = os.path.abspath("../external")
if submodule_path not in sys.path:
    sys.path.insert(0, submodule_path)

import json
import re
import torch
import numpy as np
from collections import defaultdict
from index_sets_utils import graphs_correctly_classified
from config import DATASET_NAME, DISTANCE_MODE, CORRECTLY_CLASSIFIED_ONLY

# ---------- redundant or used by ols scripts only -> delete? ------------

def get_num_changes_all_paths(pairs, changes_dict):
    counts = []
    for i, j in pairs:
        key = f"{i},{j}"
        if key in changes_dict:
            counts.append(len(changes_dict[key]))
        elif f"{j},{i}" in changes_dict:  # in case direction was flipped
            counts.append(len(changes_dict[f"{j},{i}"]))
        else:
            continue
    return counts


def count_paths_by_num_flips(idx_pair_set, flips_input_path, output_path=None, same_class=False):
    """
    For a given set of index pairs, count how many paths have 0, 1, 2, ... flips.
<
    Args:
        idx_pair_set (set of tuples): Set of (i, j) graph index pairs to consider.
        flips_input_path (str): Path to JSON file with flip data like {"i,j": [[step, label], ...], ...}
        output_path (str, optional): Where to save the resulting histogram (JSON). If None, don't save.
        same_class (boolean): If the given idx_pair_set is of same_class or diff_class category.

    Returns:
        dict: {num_flips: count} showing how many paths have that many flips.
    """
    same_class_odd_flips = []
    diff_class_even_flips = []

    # load flip data
    with open(flips_input_path) as f:
        flips_per_path = json.load(f)

    # initialize histogram
    flip_histogram = defaultdict(int)

    # count paths per number of flips
    for pair_str, flips in flips_per_path.items():
        i, j = map(int, pair_str.split(","))

        if (i, j) not in idx_pair_set and (j, i) not in idx_pair_set:
            continue

        num_flips = len(flips)

        # todo: this can probably be deleted as it was only for testing purposes
        if same_class and num_flips % 2 == 1:
            same_class_odd_flips.append((i, j))

        if not same_class and num_flips % 2 == 0:
            diff_class_even_flips.append((i, j))

        flip_histogram[num_flips] += 1

    # optionally save
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(dict(flip_histogram), f, indent=2)
    if same_class:
        with open(f"data/{DATASET_NAME}/test/same_class_odd_flips.json", "w") as f:
            json.dump(same_class_odd_flips, f, indent=2)
    if not same_class:
        with open(f"data/{DATASET_NAME}/test/diff_class_even_flips.json", "w") as f:
            json.dump(diff_class_even_flips, f, indent=2)

    return dict(flip_histogram)


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


def get_abs_flips_per_decile(idx_pairs_set, input_dir, output_dir=None, output_fname=None, distance_mode="cumulative_cost"):

    """
    Counts class changes per decile across all edit path sequences with predictions.

    Args:
        :param input_dir: Directory with .pt files containing prediction-augmented PyG graphs.
        :param output_dir: Directory to save dictionary of number of classification changes per edit step to.
        :param output_fname: Name of file to save to.

    Returns:
        dict: Mapping edit step → number of class changes at that step.

    """
    # load data for a measure on total path weighting, per-cost function or per-operation count
    if distance_mode == "cost":
        # load precalculated dict (i,j) -> edit distance
        with open(f"data/{DATASET_NAME}/analysis/distances/{DATASET_NAME}_dist_per_path.json") as f:
            distances = json.load(f)
    elif distance_mode == "num_ops":
        # load precalculated dict (i,j) -> number of operations
        with open(f"data/{DATASET_NAME}/analysis/distances/{DATASET_NAME}_num ops_per_path.json") as f:
            num_ops = json.load(f)
    else:
        print(f"[warn] given param for distance_mode or config.DISTANCE_MODE /default) does have unexpected "
              f"value {DISTANCE_MODE}. 'cost' or 'num_ops' expected. Assuming 'cost'.")
        with open(f"data/{DATASET_NAME}/analysis/distances/{DATASET_NAME}_dist_per_path.json") as f:
            distances = json.load(f)

    # initialize dict to store number of class changes per decile
    class_changes_per_decile = {decile: 0 for decile in range(10)}

    pattern = re.compile(r"g(\d+)_to_g(\d+)_it\d+_graph_sequence\.pt")

    for fname in os.listdir(input_dir):

        if not fname.endswith(".pt"):
            continue

        match = pattern.match(fname)
        if not match:
            continue

        i, j = int(match.group(1)), int(match.group(2))

        # filter for graph pairs from the given index set only
        if (i, j) not in idx_pairs_set and (j, i) not in idx_pairs_set:
            continue

        # load sequence of graphs between i, j with their predictions
        filepath = os.path.join(input_dir, fname)
        sequence = torch.load(filepath, weights_only=False)
        prev_pred = None

        # loop through sequence of predicted graphs and track prediction changes
        for step, g in enumerate(sequence):

            pred = getattr(g, "prediction", None)
            if pred is None:
                print(f"Graph without prediction at edit step {g.edit_step} in {fname}")
                continue

            if prev_pred is not None and pred != prev_pred:

                if distance_mode == "cost":
                    rel_step = g.cumulative_cost/distances[f"{i},{j}"]
                elif distance_mode == "num_ops":
                    rel_step = g.edit_step/num_ops[f"{i},{j}"]  # todo: check if this works
                else:
                    print(f"[warn] given param for distance_mode or config.DISTANCE_MODE (default) has unexpected "
                          f"value {distance_mode}. Expected 'cost' or 'num_ops'. Assuming 'cost'.")
                    rel_step = g.cumulative_cost / distances[f"{i},{j}"]
                decile = int(min(rel_step * 10, 9))
                class_changes_per_decile[decile] += 1

            prev_pred = pred

    # make keys strings
    serializable_dict = {str(k): v for k, v in class_changes_per_decile.items()}

    # optionally save dict
    if output_dir is not None and output_fname is not None:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, output_fname), "w") as f:
            json.dump(serializable_dict, f, indent=2)

    return serializable_dict


#this seems to be the same as count_paths_by_num_flips()
def get_paths_per_num_changes(input_path, output_path, index_set=None):

    # load flip history per path
    with open(input_path, "r") as f:
        flips_per_path = json.load(f)

    d = defaultdict(list)

    for path, flips in flips_per_path.items():

        if index_set is not None:
            if (i, j) not in index_set and (j, i) not in index_set:
                continue

        num_flips = len(flips)
        i, j = path.split(",")
        i = int(i)
        j = int(j)
        d[num_flips].append((i, j))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(d, f, indent=2)

    return d

# edit step analysis probably not done any further
# todo: why output dir and fname given? merge to one arg? see below for solution for earlier assumed problem
def get_abs_flips_per_edit_step(idx_pairs_set, input_dir, output_dir=None, output_fname=None):
    """
    Counts class changes at each edit step across all edit path sequences with predictions.

    Args:
        :param input_dir: Directory with .pt files containing prediction-augmented PyG graphs.
        :param output_dir: Directory to save dictionary of number of classification changes per edit step to.
        :param output_fname: Name of file to save to.

    Returns:
        dict: Mapping edit step → number of class changes at that step.

    """

    class_changes_per_step = defaultdict(int)

    pattern = re.compile(r"g(\d+)_to_g(\d+)_it\d+_graph_sequence\.pt")

    for fname in os.listdir(input_dir):

        if not fname.endswith(".pt"):
            continue

        match = pattern.match(fname)
        if not match:
            continue

        # filter for graph pairs from the index set only (same/diff class or test/train split)
        i, j = int(match.group(1)), int(match.group(2))

        if (i, j) not in idx_pairs_set and (j, i) not in idx_pairs_set:
            continue

        # load sequence of graphs with their predictions
        filepath = os.path.join(input_dir, fname)
        sequence = torch.load(filepath, weights_only=False)
        prev_pred = None

        # loop through sequence of predicted graphs and track predictions changes
        for step, g in enumerate(sequence):

            pred = getattr(g, "prediction", None)
            if pred is None:
                print(f"Graph without prediction at edit step {g.edit_step} in {fname}")
                continue

            if prev_pred is not None and pred != prev_pred:
                class_changes_per_step[g.edit_step] += 1

            prev_pred = pred

    changes_dict = dict(class_changes_per_step)

    # todo: delete? what was serialization for again?
    # serializable_dict = {str(k): v for k, v in changes_dict.items()}  # to save with json

    # optionally save dict
    if output_dir is not None and output_fname is not None:
        os.makedirs(output_dir, exist_ok=True)
        # todo: i believe this can be done via
        #   os.makedirs(os.path.dirname(save_path), exist_ok=True) for one single output arg "save_path", so no two args needed
        with open(os.path.join(output_dir, output_fname), "w") as f:
            json.dump(changes_dict, f, indent=2, sort_keys=True)

    return changes_dict
