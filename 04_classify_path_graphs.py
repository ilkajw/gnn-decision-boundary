import os
import json
import torch
from torch_geometric.datasets import TUDataset
from torch.serialization import add_safe_globals
from torch_geometric.data import Data
from GAT import GAT
from config import *


# --- config ---
model_path = f"{MODEL_DIR}/{MODEL}_model.pt"
graph_seq_dir = LEGACY_PYG_SEQ_DIR  # todo: later back to f"{ROOT}/{DATASET_NAME}/pyg_edit_path_graphs"
# for json summary and graph seqs with predictions added
# (further subdirectory 'edit_path_graphs_with_predictions' will be created for the latter in function)
output_dir = PREDICTIONS_DIR
output_fname = f"{DATASET_NAME}_{MODEL}_edit_path_predictions.json"


# --- function definition ---
def edit_path_predictions(
        model_class,
        model_kwargs,
        dataset_name,
        model_path,
        input_dir,
        output_dir,
        output_fname):
    """
    Loads all pyg graph sequences, each indexed by (source, target graph, iteration).
    Runs predictions on all graphs per sequence.
    Saves graph sequence with per-graph prediction as graph metadata to file.

    Args:
        model_path (str): Path to the saved GAT model.
        input_dir
        output_dir
        dataset_name (TUDataset): The dataset to determine feature dimension.

    Returns:
        list: A list of prediction dictionaries with metadata.
    """
    # set model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TUDataset(root=ROOT, name=dataset_name)
    model = model_class(
        in_channels=dataset.num_features,
        **model_kwargs
    ).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "edit_path_graphs_with_predictions"), exist_ok=True)
    predictions = []

    # loop through ep graph sequences indexed by source, target, iteration
    for filename in os.listdir(input_dir):

        if not filename.endswith(".pt"):
            continue

        # load ep graph sequence
        path = os.path.join(input_dir, filename)
        add_safe_globals([Data])
        graph_sequence = torch.load(path, weights_only=False)

        updated_sequence = []

        # predictions for each graph in sequence
        for graph in graph_sequence:
            graph = graph.to(device)
            with torch.no_grad():
                out = model(graph.x, graph.edge_index, torch.zeros(graph.num_nodes, dtype=torch.long).to(device))
                prob = torch.sigmoid(out.view(-1)).item()
                pred = int(prob > 0.5)

            # add prediction and probability to graph metadata
            graph.prediction = pred
            graph.probability = prob

            updated_sequence.append(graph)

            # update dict with all predictions
            predictions.append({
                "file": filename,
                "source_idx": int(getattr(graph, 'source_idx', -1)),
                "target_idx": int(getattr(graph, 'target_idx', -1)),
                "iteration": int(getattr(graph, 'iteration', -1)),
                "edit_step": int(getattr(graph, 'edit_step', -1)),
                "prediction": pred,
                "probability": prob
            })

        # save predictions as metadata for graphs
        torch.save(updated_sequence, os.path.join(output_dir, "edit_path_graphs_with_predictions", filename))
        print(f"Saved preds for file {filename} ")

    # save predictions dict
    with open(os.path.join(output_dir, output_fname), "w") as f:
        json.dump(predictions, f, indent=2)

    return predictions


# --- run ---
if __name__ == "__main__":

    for p in [model_path, graph_seq_dir]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing input directory: {p}")

    os.makedirs(output_dir, exist_ok=True)

    pred_dict = edit_path_predictions(
            model_class=MODEL_CLS,
            model_kwargs=MODEL_KWARGS,
            dataset_name=DATASET_NAME,
            model_path=model_path,
            input_dir=graph_seq_dir,
            output_dir=output_dir,
            output_fname=output_fname)
