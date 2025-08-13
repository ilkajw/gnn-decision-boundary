import sys
import os

# add submodule root to Python path
submodule_path = os.path.abspath("external")
if submodule_path not in sys.path:
    sys.path.insert(0, submodule_path)

import json
import re
import torch
import numpy as np
from collections import defaultdict
from config import DATASET_NAME
from pg_gnn_edit_paths.utils.io import load_edit_paths_from_file


def get_distance_per_path(input_path=f"external/pg_gnn_edit_paths/example_paths_{DATASET_NAME}",
                          output_path=f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_dist_per_pair.json"):

    edit_paths = load_edit_paths_from_file(db_name=DATASET_NAME,
                                           file_path=input_path)
    distances = {}
    for (i, j), path_list in edit_paths.items():
        if path_list:
            distances[f"{i},{j}"] = path_list[0].distance
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(distances, f, indent=2)

    return distances


def get_num_ops_per_path(input_path=f"external/pg_gnn_edit_paths/example_paths_{DATASET_NAME}",
                         output_path=f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_num_ops_per_pair.json"):

    edit_paths = load_edit_paths_from_file(db_name=DATASET_NAME,
                                           file_path=input_path)
    num_ops = {}
    for (i, j), path_list in edit_paths.items():
        if path_list:
            num_ops[f"{i},{j}"] = len(path_list[0].all_operations) if len(path_list[0].all_operations) > 0 else 1
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(num_ops, f, indent=2)

    return num_ops


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
    #serializable_dict = {str(k): v for k, v in changes_dict.items()}  # to save with json

    # optionally save dict
    if output_dir is not None and output_fname is not None:
        os.makedirs(output_dir, exist_ok=True)
        # todo: i believe this can be done via
        #   os.makedirs(os.path.dirname(save_path), exist_ok=True) for one single output arg "save_path", so no two args needed
        with open(os.path.join(output_dir, output_fname), "w") as f:
            json.dump(changes_dict, f, indent=2, sort_keys=True)

    return changes_dict



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
    if distance_mode == "cost_function":
        # load precalculated dict (i,j) -> edit distance
        with open(f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_dist_per_pair.json") as f:
            distances = json.load(f)
    elif distance_mode == "operations_count":
        # load precalculated dict (i,j) -> number of operations
        with open(f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_num ops_per_pair.json") as f:
            num_ops = json.load(f)
    else:
        print("Choose a valid param for distance_mode: 'cost_function' or 'operations_count'")
        return

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

                if distance_mode == "cost_function":
                    rel_step = g.cumulative_cost/distances[f"{i},{j}"]
                else:
                    rel_step = g.edit_step/num_ops[f"{i},{j}"]  # todo: check if this works

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


def flip_distribution_by_indexset(idx_pair_set, dist_input_path, flips_input_path, output_path=None):

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
        if (i, j) not in idx_pair_set and (j, i) not in idx_pair_set:
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


def count_paths_by_num_flips(idx_pair_set, flips_input_path, output_path=None, same_class = False):
    """
    For a given set of index pairs, count how many paths have 0, 1, 2, ... flips.

    Args:
        idx_pair_set (set of tuples): Set of (i, j) graph index pairs to consider.
        flips_input_path (str): Path to JSON file with flip data like {"i,j": [[step, label], ...], ...}
        output_path (str, optional): Where to save the resulting histogram (JSON). If None, don't save.

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
        with open("data/MUTAG/test/same_class_odd_flips.json", "w") as f:
            json.dump(same_class_odd_flips, f, indent=2)
    if not same_class:
        with open("data/MUTAG/test/diff_class_even_flips.json", "w") as f:
            json.dump(diff_class_even_flips, f, indent=2)

    return dict(flip_histogram)


def get_flip_steps_per_pair(input_dir, output_dir=None, output_fname=None, verbose=False):

    """
    Tracks classification changes along edit paths for each graph pair.

    Returns a dictionary:
        (i, j) → list of (edit_step, new_class)

    Args:
        :param input_dir: Directory with .pt files containing prediction-augmented PyG graphs.
        :param verbose: If True, print progress.
        :param output_dir: Directory to save dictionary of classification changes per sequence to.
        :param output_fname: Name of file to save to.
    """

    pattern = re.compile(r"g(\d+)_to_g(\d+)_it\d+_graph_sequence\.pt")
    changes_dict = {}

    for fname in os.listdir(input_dir):
        if not fname.endswith(".pt"):
            continue

        # extract indices from existing files of predicted graph sequences
        match = pattern.match(fname)
        if not match:
            continue

        i, j = int(match.group(1)), int(match.group(2))
        filepath = os.path.join(input_dir, fname)

        sequence = torch.load(filepath, weights_only=False)
        prev_pred = None
        change_steps = []

        # loop through each sequence from i to j and track at which steps changes happen to which class
        for step, g in enumerate(sequence):

            if not hasattr(g, "prediction"):
                print(f"Missing prediction of graph at edit step {g.edit_step} in file {fname}")
                continue

            pred = getattr(g, "prediction", None)

            if prev_pred is not None and pred != prev_pred:
                change_steps.append((g.edit_step, pred))

            prev_pred = pred

            changes_dict[(i, j)] = change_steps

        if verbose:
            print(f"Processed: {fname} | Changes: {change_steps}")

    serializable_dict = {f"{i},{j}": val for (i, j), val in changes_dict.items()}

    # optionally save dict
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, output_fname), "w") as f:
            json.dump(serializable_dict, f, indent=2)
        print("Saved changes per pair dict")

    return serializable_dict


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


def get_rel_flips_per_decile_by_k(
    max_k,
    dist_input_path,
    flips_input_path,
    idx_pair_set=None,
    output_dir=None,
    output_prefix=None,
    include_paths=False,
):
    # load
    with open(dist_input_path, "r") as f:
        distances = json.load(f)
    with open(flips_input_path, "r") as f:
        flips_per_path = json.load(f)

    def get_distance(i, j):
        s1, s2 = f"{i},{j}", f"{j},{i}"
        return distances.get(s1, distances.get(s2))

    # accumulators
    abs_counts_by_k = {k: [0]*10 for k in range(1, max_k+1)}
    paths_by_k = {k: [] for k in range(1, max_k+1)} if include_paths else None
    num_paths_by_k = defaultdict(int)

    for pair_str, flips in flips_per_path.items():
        if not flips:
            continue
        i, j = map(int, pair_str.split(","))
        if idx_pair_set is not None and (i, j) not in idx_pair_set and (j, i) not in idx_pair_set:
            continue

        k = len(flips)
        if k < 1 or k > max_k:
            continue

        dist = get_distance(i, j)
        if not dist:  # None or 0
            continue

        # per-path decile counts
        decile_counts = [0]*10
        for step, _lbl in flips:
            rel = step / dist
            d = int(min(rel * 10, 9))  # bin 0..9, clamp 1.0 to 9
            decile_counts[d] += 1

        total_flips = sum(decile_counts)
        if total_flips == 0:
            continue

        # accumulate absolutes
        acc = abs_counts_by_k[k]
        for d in range(10):
            acc[d] += decile_counts[d]

        num_paths_by_k[k] += 1
        if include_paths:
            paths_by_k[k].append({"pair": [i, j], "flips": flips})

    # build result
    result = {}
    for k in range(1, max_k+1):
        abs_counts = abs_counts_by_k[k]
        total_abs = sum(abs_counts)
        if total_abs > 0:
            global_prop = [c / total_abs for c in abs_counts]
        else:
            global_prop = [0.0]*10

        # store relative and absolute values per-k
        entry = {
            "num_paths": int(num_paths_by_k[k]),
            "avg_proportion": {str(d): float(global_prop[d]) for d in range(10)},
            "abs_counts": {str(d): int(abs_counts[d]) for d in range(10)},
            "abs_distribution": {str(d): int(abs_counts[d]) for d in range(10)},
        }
        # optionally add set of contributing paths
        if include_paths:
            entry["paths"] = paths_by_k[k]
        result[str(k)] = entry

    if output_dir and output_prefix:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{output_prefix}_per_k_deciles.json"), "w") as f:
            json.dump(result, f, indent=2)

    return result


