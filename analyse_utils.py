import sys
import os

# add submodule root to Python path
submodule_path = os.path.abspath("external")
if submodule_path not in sys.path:
    sys.path.insert(0, submodule_path)

import json
import re
import torch
from collections import defaultdict
from config import DATASET_NAME
from pg_gnn_edit_paths.utils.io import load_edit_paths_from_file


def get_distance_per_pair(input_path=f"external/pg_gnn_edit_paths/example_paths_{DATASET_NAME}",
                          output_path=f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_dist_per_pair.json"):

    edit_paths = load_edit_paths_from_file(db_name=DATASET_NAME,
                                           file_path=input_path)
    distances = {}

    for (i, j), path_list in edit_paths.items():

        if path_list:
            distances[f"{i},{j}"] = path_list[0].distance  # or average/multiple if needed

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(distances, f, indent=2)

    return distances


def get_class_changes_per_edit_step(idx_pairs_set, input_dir, output_dir=None, output_fname=None):

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
    serializable_dict = {str(k): v for k, v in changes_dict.items()}  # to save with json

    # optionally save dict
    if output_dir is not None and output_fname is not None:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, output_fname), "w") as f:
            json.dump(serializable_dict, f, indent=2)

    return serializable_dict


def get_class_changes_per_decile(idx_pairs_set, input_dir, output_dir=None, output_fname=None):

    """
    Counts class changes at each edit step across all edit path sequences with predictions.

    Args:
        :param input_dir: Directory with .pt files containing prediction-augmented PyG graphs.
        :param output_dir: Directory to save dictionary of number of classification changes per edit step to.
        :param output_fname: Name of file to save to.

    Returns:
        dict: Mapping edit step → number of class changes at that step.

    """

    # load precalculated dict (i,j) -> edit distance
    with open(f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_dist_per_pair.json") as f:
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

        # loop through sequence of predicted graphs and track predictions changes
        for step, g in enumerate(sequence):

            pred = getattr(g, "prediction", None)
            if pred is None:
                print(f"Graph without prediction at edit step {g.edit_step} in {fname}")
                continue

            if prev_pred is not None and pred != prev_pred:
                rel_step = g.edit_step/distances[f"{i},{j}"]
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


def get_class_change_steps_per_pair(input_dir, output_dir=None, output_fname=None, verbose=False):

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

        if change_steps:
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


def get_num_changes_per_path(pairs, changes_dict):
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