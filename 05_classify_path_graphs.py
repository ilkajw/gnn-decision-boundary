"""
Run a saved GNN on PyG edit-path graph sequences and write predictions.

Loads a trained model, iterates over PyG graph sequences (ROOT/DATASET_NAME/pyg_edit_path_graphs),
runs the model on each graph, attaches `prediction` and `probability` attributes to each
PyG graph, saves the updated sequences under PREDICTIONS_DIR/<sequences_subdir_name>/,
and writes a JSON summary of all per-graph predictions.

Config required: DATASET_NAME, ROOT, MODEL, MODEL_CLS, MODEL_KWARGS, MODEL_DIR, PREDICTIONS_DIR.
Inputs:
  - trained model: MODEL_DIR/<DATASET_NAME>_<MODEL>_model.pt
  - PyG sequences: ROOT/DATASET_NAME/pyg_edit_path_graphs
Outputs:
  - updated sequences: PREDICTIONS_DIR/<sequences_subdir_name>/*.pt
  - predictions JSON: PREDICTIONS_DIR/<DATASET_NAME>_<MODEL>_edit_path_predictions.json
"""

import os
import json
import torch
from torch_geometric.datasets import TUDataset
from torch.serialization import add_safe_globals
from torch_geometric.data import Data

from config import ROOT, DATASET_NAME, MODEL, MODEL_CLS, MODEL_KWARGS, MODEL_DIR, PREDICTIONS_DIR, LEGACY_PYG_SEQ_DIR


# ---- Input paths ----
model_path = os.path.join(MODEL_DIR, f"{DATASET_NAME}_{MODEL}_model.pt")
graph_sequences_dir = os.path.join(ROOT, DATASET_NAME, "pyg_edit_path_graphs")

# ---- Output paths ----
output_dir = PREDICTIONS_DIR
# subdirectory will be created within output_dir for predicted graph sequences
sequences_subdir_name = 'edit_path_graphs_with_predictions'
output_fname = f"{DATASET_NAME}_{MODEL}_edit_path_predictions.json"


# --- Function definition ---
def edit_path_predictions(
        model_class,
        model_kwargs,
        dataset_name,
        model_path,
        input_dir,
        output_dir,
        sequences_subdir_name,
        output_fname,
        verbose=False
):
    """
    Loads all PyG graph sequences, each indexed by (source, target graph, iteration).
    Runs predictions on all graphs per sequence.
    Saves graph sequence with per-graph prediction as graph attributes to file.

    Args:
        model_path (str): Path to the saved GAT model.
        input_dir
        output_dir
        dataset_name (TUDataset): The dataset to determine feature dimension.

    Returns:
        list: A list of prediction dictionaries with metadata.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TUDataset(root=ROOT, name=dataset_name)
    add_safe_globals([Data])
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, sequences_subdir_name), exist_ok=True)
    predictions = []

    # Instantiate model and load weight state
    model = model_class(
        in_channels=dataset.num_features,
        **model_kwargs
    ).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Loop through graph sequences
    for filename in os.listdir(input_dir):

        if not filename.endswith(".pt"):
            continue

        path = os.path.join(input_dir, filename)
        graph_sequence = torch.load(path, weights_only=False)

        updated_sequence = []

        # Predictions for each graph in sequence
        for graph in graph_sequence:
            graph = graph.to(device)
            with torch.no_grad():
                out = model(graph.x, graph.edge_index, torch.zeros(graph.num_nodes, dtype=torch.long).to(device))
                prob = torch.sigmoid(out.view(-1)).item()
                pred = int(prob > 0.5)

            # Add prediction and probability to graph attributes
            graph.prediction = pred
            graph.probability = prob

            updated_sequence.append(graph)

            # Update dict recording all predictions
            predictions.append({
                "file": filename,
                "source_idx": int(getattr(graph, 'source_idx', -1)),
                "target_idx": int(getattr(graph, 'target_idx', -1)),
                "iteration": int(getattr(graph, 'iteration', -1)),
                "operation": getattr(graph, 'operation', None),
                "edit_step": int(getattr(graph, 'edit_step', -1)),
                "cumulative_cost": getattr(graph, 'cumulative_cost', -1),
                "prediction": pred,
                "probability": prob,
                "path_distance": int(getattr(graph, 'distance', -1)),
                "num_operations_incl_insertion": getattr(graph, "num_operations_incl_insertion", -1),
            })
        # Save updated sequence
        torch.save(updated_sequence, os.path.join(output_dir, sequences_subdir_name, filename))
        if verbose:
            print(f"[info] Saved predictions for file {filename} ")

    # Save JSON with prediction records
    with open(os.path.join(output_dir, output_fname), "w") as f:
        json.dump(predictions, f, indent=2)

    return predictions


# --- Run ---
if __name__ == "__main__":

    for p in [model_path, graph_sequences_dir]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing input directory: {p}")

    os.makedirs(output_dir, exist_ok=True)

    pred_dict = edit_path_predictions(
            model_class=MODEL_CLS,
            model_kwargs=MODEL_KWARGS,
            dataset_name=DATASET_NAME,
            model_path=model_path,
            input_dir=graph_sequences_dir,
            output_dir=output_dir,
            sequences_subdir_name=sequences_subdir_name,
            output_fname=output_fname,
            verbose=False
    )
