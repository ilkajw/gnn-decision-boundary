# TODO: file descriptor

import os
import json
import re
import torch
from torch.serialization import add_safe_globals
from torch_geometric.data import Data

from config import DATASET_NAME, MODEL, PREDICTIONS_DIR, ROOT, MODEL_DEPENDENT_PRECALCULATIONS_DIR, \
    MODEL_INDEPENDENT_PRECALCULATIONS_DIR
from external.pg_gnn_edit_paths.utils.io import load_edit_paths_from_file


# ---- Set inputs ----
edit_paths_input_dir = os.path.join("external", "pg_gnn_edit_paths", f"example_paths_{DATASET_NAME}")
graph_sequences_input_dir = os.path.join(PREDICTIONS_DIR, "edit_path_graphs_with_predictions")

# ----- Set outputs ----
dist_cost_fname = f"{DATASET_NAME}_dist_per_path.json"
dist_num_ops_fname = f"{DATASET_NAME}_num_ops_per_path.json"
dist_output_path = os.path.join(MODEL_INDEPENDENT_PRECALCULATIONS_DIR, dist_cost_fname)
num_ops_output_path = os.path.join(MODEL_INDEPENDENT_PRECALCULATIONS_DIR, dist_num_ops_fname)

# Files going into MODEL_DEPENDENT_PRECALCULATIONS_DIR
flip_occ_edit_step_output_fname = f"{DATASET_NAME}_{MODEL}_flip_occurrences_per_path_by_edit_step.json"
flip_occ_cost_output_fname = f"{DATASET_NAME}_{MODEL}_flip_occurrences_per_path_by_cost.json"


# ---- Function definitions ----
def cost_distance_per_path(input_path, output_path):

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


def num_operations_per_path(
        sequences_dir=os.path.join(ROOT, DATASET_NAME, "pyg_edit_path_graphs"),
        output_path=os.path.join(ROOT, DATASET_NAME, "precalculations")
):
    # TODO: docstring
    add_safe_globals([Data])
    pattern = re.compile(r"g(\d+)_to_g(\d+)_it\d+_graph_sequence\.pt")
    num_operations = {}

    for fname in sorted(os.listdir(sequences_dir)):

        if not fname.endswith(".pt"):
            continue

        # extract indices from existing files of predicted graph sequences
        match = pattern.match(fname)
        if not match:
            continue

        i, j = int(match.group(1)), int(match.group(2))

        filepath = os.path.join(sequences_dir, fname)
        sequence = torch.load(filepath, map_location="cpu", weights_only=False)
        if not isinstance(sequence, (list, tuple)) or len(sequence) == 0:
            raise RuntimeError(f"Loaded sequence is empty or not list/tuple: {fname}")
        first = sequence[0]
        try:
            val = first["num_operations_incl_insertion"]
        except Exception as e:
            raise KeyError(f"'num_operations_incl_insertion' missing in first element of {fname}") from e
        if not isinstance(val, (int, float)):
            raise TypeError(f"'num_operations_incl_insertion' is not numeric in {fname}: {type(val)}")
        num_operations[f"{i},{j}"] = int(val)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(num_operations, f, indent=2)

    return num_operations


def flip_occurrences_per_path_edit_step(input_dir, output_dir=None, output_fname=None, verbose=False):

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
    add_safe_globals([Data])
    pattern = re.compile(r"g(\d+)_to_g(\d+)_it\d+_graph_sequence\.pt")
    changes_dict = {}

    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".pt"):
            continue

        # extract indices from existing files of predicted graph sequences
        match = pattern.match(fname)
        if not match:
            continue

        i, j = int(match.group(1)), int(match.group(2))
        filepath = os.path.join(input_dir, fname)

        sequence = torch.load(filepath, map_location="cpu", weights_only=False)
        prev_pred = None
        change_steps = []

        # loop through each sequence from i to j and track at which steps changes happen to which class
        for g in sequence:

            # Handle 38 sequences with operation null for their start graphs
            if getattr(g, "edit_step") == 0 and not hasattr(g, "operation"):
                setattr(g, "operation", "start")

            if not hasattr(g, "edit_step"):
                raise AttributeError(f"'edit_step' missing in a graph of {fname}")
            if not hasattr(g, "prediction"):
                raise AttributeError(f"'prediction' missing at edit_step={g.edit_step} in file {fname}")
            if not hasattr(g, "operation"):
                raise AttributeError(f"'operation' missing at edit_step={g.edit_step} in file {fname}")

            pred = g.prediction

            if prev_pred is not None and pred != prev_pred:
                change_steps.append((g.edit_step, pred, g.operation))

            prev_pred = pred

        changes_dict[(i, j)] = change_steps

        if verbose:
            print(f"Processed: {fname} | Changes: {change_steps}")

    serializable_dict = {f"{i},{j}": val for (i, j), val in changes_dict.items()}

    # Optionally save dictionary
    if output_dir:

        if not output_fname:
            raise ValueError("output_fname must be provided when output_dir is set.")

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_fname)
        with open(output_path, "w") as f:
            json.dump(serializable_dict, f, indent=2)

        print(f"Saved changes per path by edit step to {output_path}.")

    return serializable_dict


def flip_occurrences_per_path_cum_cost(input_dir, output_dir=None, output_fname=None, verbose=False):

    """
    Tracks classification changes along edit paths for each graph pair.

    Returns a dictionary:
        (i, j) → list of (cumulative_cost, new_class)

    Args:
        :param input_dir: Directory with .pt files containing prediction-augmented PyG graphs.
        :param verbose: If True, print progress.
        :param output_dir: Directory to save dictionary of classification changes per sequence to.
        :param output_fname: Name of file to save to.
    """
    add_safe_globals([Data])
    pattern = re.compile(r"g(\d+)_to_g(\d+)_it\d+_graph_sequence\.pt")
    changes_dict = {}

    for fname in os.listdir(input_dir):

        if not fname.endswith(".pt"):
            continue

        # Extract indices from existing files of predicted graph sequences
        match = pattern.match(fname)
        if not match:
            continue

        i, j = int(match.group(1)), int(match.group(2))
        filepath = os.path.join(input_dir, fname)

        sequence = torch.load(filepath, map_location="cpu", weights_only=False)
        prev_pred = None
        change_steps = []

        # loop through each sequence from i to j and track at which steps changes happen to which class
        for g in sequence:

            if not hasattr(g, "cumulative_cost"):
                raise AttributeError(
                    f"'cumulative_cost' missing at edit_step={getattr(g, 'edit_step', '?')} in {fname}")
            if not hasattr(g, "prediction"):
                raise AttributeError(f"'prediction' missing at edit_step={getattr(g, 'edit_step', '?')} in {fname}")
            if not hasattr(g, "operation"):
                raise AttributeError(f"'operation' missing at edit_step={getattr(g, 'edit_step', '?')} in {fname}")

            pred = g.prediction

            if prev_pred is not None and pred != prev_pred:
                change_steps.append((g.cumulative_cost, pred, g.operation))

            prev_pred = pred

        changes_dict[(i, j)] = change_steps

        if verbose:
            print(f"Processed: {fname} | Changes: {change_steps}")

    serializable_dict = {f"{i},{j}": val for (i, j), val in changes_dict.items()}

    # optionally save dict
    if output_dir:

        if not output_fname:
            raise ValueError("output_fname must be provided when output_dir is set.")

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_fname)
        with open(output_path, "w") as f:
            json.dump(serializable_dict, f, indent=2)

        print(f"Saved changes per path by cost to {output_path}.")

    return serializable_dict


if __name__ == "__main__":

    for p in [edit_paths_input_dir, graph_sequences_input_dir]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing input directory: {p}")

    os.makedirs(MODEL_DEPENDENT_PRECALCULATIONS_DIR, exist_ok=True)
    os.makedirs(MODEL_INDEPENDENT_PRECALCULATIONS_DIR, exist_ok=True)

    cost_distance_per_path(input_path=edit_paths_input_dir,
                           output_path=dist_output_path)

    num_operations_per_path(sequences_dir=graph_sequences_input_dir,
                            output_path=num_ops_output_path)

    flip_occurrences_per_path_edit_step(
        input_dir=graph_sequences_input_dir,
        output_dir=MODEL_DEPENDENT_PRECALCULATIONS_DIR,
        output_fname=flip_occ_edit_step_output_fname)

    flip_occurrences_per_path_cum_cost(
        input_dir=graph_sequences_input_dir,
        output_dir=MODEL_DEPENDENT_PRECALCULATIONS_DIR,
        output_fname=flip_occ_cost_output_fname)
