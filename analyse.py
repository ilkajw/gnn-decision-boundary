import json
import os
from itertools import combinations
from model import GAT
from config import ROOT, DATASET_NAME, HIDDEN_CHANNELS, HEADS, DROPOUT
from torch_geometric.datasets import TUDataset
import torch
from config import *


def correct_class_idxs():

    """Returns the indices of all graphs classified correctly by our GAT model, grouped by all correct
    classification and per-class classifications."""

    with open("data/predictions/mutag_predictions.json") as f:
        predictions = json.load(f)

    # get correctly classified graph indices
    correct_idxs = [int(i) for i, entry in predictions.items() if entry["correct"]]

    # get correctly classified class 1 graph indices
    correct_class_1_idxs = [int(i) for i, entry in predictions.items()
                            if entry["correct"] and entry["true_label"] == 1]

    # get correctly classified class 0 graph indices
    correct_class_0_idxs = [int(i) for i, entry in predictions.items()
                            if entry["correct"] and entry["true_label"] == 0]

    return correct_idxs, correct_class_0_idxs, correct_class_1_idxs


def valid_idxs():

    all, class0, class1 = correct_class_idxs()

    # generate all (i, j) pairs from class 0 and class 1
    same = list(combinations(class0, 2)) + list(combinations(class1, 2))

    # generate all (i, j) pairs where one is from class 0 and one from class 1
    diff = [(i, j) for i in class0 for j in class1]  # todo: other way necessary?

    return same, diff


def graphs_by_edit_step(graph_idx_pairs, step):
    collected_graphs = []

    for i, j in graph_idx_pairs:
        file_path = os.path.join("data/edit_path_graphs", f"g{i}_to_g{j}_sequence.pt")

        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found.")
            continue

        graph_sequence = torch.load(file_path)

        if step - 1 < len(graph_sequence):  # step is 1-based
            collected_graphs.append(graph_sequence[step - 1], i, j)
        else:
            print(f"Step {step} exceeds length of sequence in g{i}_to_g{j}")

    return collected_graphs

# todo: check if really want to hand graph pairs to function or if not calc for all graphs, then save data
#  with result and filter later on
# todo: but keep indexes for correctly classified maybe!
def predict_leg_edit_graphs(graph_pairs):

    model_path = "model/model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("model/best_split.json", "r") as f:
        best_split = json.load(f)
    train_idxs = set(best_split["train_idx"])
    test_idxs = set(best_split["test_idx"])

    # load trained model
    model = GAT(
        in_channels=NUM_FEATURES,
        hidden_channels=HIDDEN_CHANNELS,
        heads=HEADS,
        dropout=DROPOUT
    ).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # load true labels for source, target node from predictions file
    with open("data/predictions/mutag_predictions.json") as f:
        predictions = json.load(f)

    results = []

    for graph, i, j in graph_pairs:
        graph = graph.to(device)
        with torch.no_grad():
            out = model(graph)
            prob = torch.sigmoid(out.view(-1))
            pred = (prob > 0.5).long().item()

        results.append({
            "prediction": pred,
            "probability": prob.item(),
            "edit_step": graph['edit_step'],
            "source_idx": i,
            "target_idx": j,
            "source_class": predictions[str(i)]["true_label"],  # todo: potentially redundant if we only look at correctly classified graphs
            "target_class": predictions[str(j)]["true_label"],
            "correct_source": predictions[str(i)]["correct"],
            "correct_target": predictions[str(j)]["correct"],
            "source_in_train": i in best_split["train_idx"],
            "target_in_train": i in best_split["train_idx"]

            # todo: potentially it is enough to filter back here and not filter for index sets before as we can filter
            # through the results dict later on for graphs of interest. potentially also encode train/test split of
            # sorce/target
        })

    return results
