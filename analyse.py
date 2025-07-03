import json
from itertools import combinations
from model import GAT
import torch
from config import *


# todo: rewrite to dictionary "index: class"
def correct_class_idxs():

    """Returns the indices of all MUTAG graphs classified correctly by our GAT model, grouped by all correct
    classifications and correct per-class classifications."""

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


def idx_set_same_class():

    """Creates a lists of MUTAG graph index pairs being from the same class."""

    class0, class1 = correct_class_idxs()[1, 2] # todo: indexing correct?

    # generate all (i, j) pairs from class 0 and class 1
    same = list(combinations(class0, 2)) + list(combinations(class1, 2))

    return same

def idx_set_diff_class():

    """Creates a lists of all MUTAG graph index pairs being from different classes."""

    class0, class1 = correct_class_idxs()[1, 2]

    # generate all (i, j) pairs where one is from class 0 and one from class 1
    diff = [(i, j) for i in class0 for j in class1]  # todo: other way necessary?

    return diff




# todo: check if really want to hand graph pairs to function or if not calc for all graphs, then save data
#  with result and filter later on
# todo: but keep indexes to insert correctly classified maybe!

# todo: has to change due to florians implementation. check metadata of florians impl. potentially add method
#  "add_metadata" for things like train/test split
def predict_log_edit_graphs(graph_pairs):

    # todo: why graph pairs??

    model_path = "model/model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model training and test split for
    with open("model/best_split.json", "r") as f:
        best_split = json.load(f)

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
