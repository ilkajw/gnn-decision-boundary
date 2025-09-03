import os
import json
import re
import torch
from torch.serialization import add_safe_globals
from torch_geometric.data import Data

from config import DATASET_NAME, MODEL, PREDICTIONS_DIR, ANALYSIS_DIR
from external.pg_gnn_edit_paths.utils.io import load_edit_paths_from_file


# ---- set input, output paths ----
edit_paths_input_dir = f"external/pg_gnn_edit_paths/example_paths_{DATASET_NAME}"
pred_graph_seq_input_dir = f"{PREDICTIONS_DIR}/edit_path_graphs_with_predictions"
pred_graph_seq_cum_cost_input_dir = f"{PREDICTIONS_DIR}/edit_path_graphs_with_predictions_CUMULATIVE_COST"
# in case of overwriting: pred_graph_seq_cum_cost_input_dir = pred_graph_seq_input_dir

output_dir = ANALYSIS_DIR
dist_cost_fname = f"{DATASET_NAME}_dist_per_path.json"
dist_num_ops_fname = f"{DATASET_NAME}_num_ops_per_path.json"
flip_occ_edit_step_output_fname = f"{DATASET_NAME}_{MODEL}_flip_occurrences_per_path_by_edit_step.json"
flip_occ_cost_output_fname = f"{DATASET_NAME}_{MODEL}_flip_occurrences_per_path_by_cost.json"


# ---- helpers ----
def get_distance_per_path(input_path, output_path):

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


def get_num_ops_per_path(input_path, output_path):

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

# todo: merge following two functions into one with "by_cost=True/False"


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
        print("Saved changes per path by edit step.")

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
        for step, g in enumerate(sequence):

            if not hasattr(g, "prediction"):
                print(f"Missing prediction of graph at edit step {g.edit_step} in file {fname}")
                continue

            pred = getattr(g, "prediction", None)

            if prev_pred is not None and pred != prev_pred:
                change_steps.append((g.cumulative_cost, pred))

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
        print("Saved changes per path by cost.")

    return serializable_dict


if __name__ == "__main__":

    for p in [edit_paths_input_dir, pred_graph_seq_input_dir, pred_graph_seq_cum_cost_input_dir]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing input directory: {p}")

    os.makedirs(output_dir, exist_ok=True)

    cost_output_path = os.path.join(output_dir, dist_cost_fname)
    num_ops_output_path = os.path.join(output_dir, dist_num_ops_fname)
    get_distance_per_path(input_path=edit_paths_input_dir,
                          output_path=cost_output_path)

    get_num_ops_per_path(input_path=edit_paths_input_dir,
                         output_path=num_ops_output_path)

    flip_occurrences_per_path_edit_step(
        input_dir=pred_graph_seq_input_dir,
        output_dir=output_dir,
        output_fname=flip_occ_edit_step_output_fname)

    flip_occurrences_per_path_cum_cost(
        input_dir=pred_graph_seq_cum_cost_input_dir,
        output_dir=output_dir,
        output_fname=flip_occ_cost_output_fname)
