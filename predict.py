import json

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from config import *
from model import GAT
from edit_path_graphs_old import *
import pickle


def mutag_predictions():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load dataset
    dataset = TUDataset(root=ROOT, name=DATASET_NAME)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)  # batch_size=1 to keep index tracking

    # re-instantiate model
    model = GAT(
        in_channels=dataset.num_features,
        hidden_channels=HIDDEN_CHANNELS,
        heads=HEADS,
        dropout=DROPOUT
    ).to(device)
    model.load_state_dict(torch.load("model/model.pt"))
    model.eval()

    predictions = {}
    correct_class_0 = {}
    correct_class_1 = {}

    for idx, data in enumerate(loader):
        data = data.to(device)
        out = model(data)
        probs = torch.sigmoid(out.view(-1))
        pred = (probs > 0.5).long()
        true = data.y.item()
        correct = int(pred == true)

        predictions[idx] = {
            'true_label': true,
            'pred_label': pred,
            'correct': correct
        }
        # separate correctly classified graphs
        if correct:
            if true == 0:
                correct_class_0[idx] = predictions[idx]
            elif true == 1:
                correct_class_1[idx] = predictions[idx]

    os.makedirs("data/predictions", exist_ok=True)

    with open("data/predictions/mutag_predictions.json", "wb") as f:
        pickle.dump(predictions, f)


def edit_path_predictions(model_path, input_dir, output_dir, dataset_name):
    """
    Loads all pyg graph sequences, each indexed by source, target graph and iteration.
    Runs predictions on graphs per sequence.
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
    dataset = TUDataset(root="data", name=dataset_name)
    model = GAT(
        in_channels=dataset.num_features,
        hidden_channels=HIDDEN_CHANNELS,
        heads=HEADS,
        dropout=DROPOUT
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    predictions = []

    # loop through ep graph sequences indexed by source, target, iteration
    for filename in os.listdir(input_dir):

        if not filename.endswith(".pt"):
            continue

        # load ep graph sequence
        path = os.path.join(input_dir, filename)
        graph_sequence = torch.load(path)

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

            # dict with all predictions
            predictions.append({
                "file": filename,
                "source_idx": int(getattr(graph, 'source_idx', -1)),
                "target_idx": int(getattr(graph, 'target_idx', -1)),
                "iteration": int(getattr(graph, 'iteration', -1)),
                "edit_step": int(getattr(graph, 'edit_step', -1)),
                "prediction": pred,
                "probability": prob
            })

        # todo: rethink if putting pred to metadata and saving is useful or if pediction dict is enough for analysis
        torch.save(updated_sequence, os.path.join(output_dir, filename))

    return predictions


def add_metadata_to_edit_path_predictions(pred_dict, base_pred_path, split_path, output_path):
    """
    Enriches dictionary entries of edit path predictions with additional metadata:
    - true class labels of source and target
    - whether source/target are in training split
    - whether source/target were classified correctly

    Args:
        :param split_path: Path to saved train, test split
        :param base_pred_path: Path to original MUTAG predictions file
        :param pred_dict: Predictions (list): List of prediction dicts on edit path graphs
        :param output_path: Path to file where enriched dictionary is saved

    Returns:
        list: Enriched prediction dictionaries
    """

    # load train, test split
    with open(split_path, "r") as f:
        split = json.load(f)

    # load predictions on org mutag graphs
    with open(base_pred_path, "r") as f:
        base_preds = json.load(f)

    # add train vs. test split, classes of source & target, correct classification of source & train to metadata
    for entry in pred_dict:

        i = str(entry["source_idx"])
        j = str(entry["target_idx"])

        entry["source_in_train"] = int(i) in split["train_idx"]
        entry["target_in_train"] = int(j) in split["train_idx"]

        entry["correct_source"] = base_preds[i]["correct"]
        entry["correct_target"] = base_preds[j]["correct"]

        entry["source_class"] = base_preds[i]["true_label"]
        entry["target_class"] = base_preds[j]["true_label"]

    with open(output_path, "w") as f:
        json.dump(pred_dict, f, indent=2)
    print(f"DEBUG: saved enriched predictions to {output_path}")
