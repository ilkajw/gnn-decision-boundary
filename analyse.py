import json
import os
import re
import torch

from collections import defaultdict


def count_class_changes_per_edit_step(input_dir, output_dir=None, output_fname=None, verbose=False):

    """
    Counts class changes at each edit step across all edit path sequences with predictions.

    Args:
    :param input_dir: Directory with .pt files containing prediction-augmented PyG graphs.
        :param verbose: If True, print progress.
        :param output_dir: Directory to save dictionary of number of classification changes per edit step to.
        :param output_fname: Name of file to save to.

    Returns:
        dict: Mapping edit step → number of class changes at that step.

    """

    class_changes_per_step = defaultdict(int)

    for fname in os.listdir(input_dir):
        if not fname.endswith(".pt"):
            continue

        filepath = os.path.join(input_dir, fname)
        sequence = torch.load(filepath, weights_only=False)
        prev_pred = None

        for step, g in enumerate(sequence):

            pred = getattr(g, "prediction", None)
            if pred is None:
                print(f"Graph without prediction at edit step {g.edit_step} in {fname}")
                continue

            if prev_pred is not None and pred != prev_pred:
                class_changes_per_step[g.edit_step] += 1

            prev_pred = pred

        if verbose:
            print(f"Processed: {fname}")

    changes_dict = dict(class_changes_per_step)
    serializable_dict = {str(k): v for k, v in changes_dict.items()}

    # optionally save dict
    if output_dir:
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

        match = pattern.match(fname)
        if not match:
            continue

        i, j = int(match.group(1)), int(match.group(2))
        filepath = os.path.join(input_dir, fname)

        sequence = torch.load(filepath, weights_only=False)
        prev_pred = None
        change_steps = []

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

    return serializable_dict
