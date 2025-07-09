import itertools
import json
import os


def graphs_correctly_classified(dataset_name):

    """Returns the indices of all MUTAG graphs classified correctly by our GAT model."""

    with open(f"data/{dataset_name}/predictions/{dataset_name}_predictions.json") as f:
        predictions = json.load(f)

    correct_idxs = [int(i) for i, entry in predictions.items() if entry["correct"]]

    return correct_idxs


def graph_index_pairs_same_class(dataset_name,
                                 correctly_classified_only=True,
                                 save_path=None):
    """
    Returns all index pairs (i, j) from MUTAG where both graphs are of the same class.
    If correctly_classified_only is True, only include graphs correctly classified by the model.
    Optionally saves the result to a JSON file.
    """

    # read in predictions of our model on MUTAG graphs
    with open(f"data/{dataset_name}/predictions/{dataset_name}_predictions.json") as f:
        predictions = json.load(f)

    # filter graph indexes, if only correctly classified graphs should be considered
    if correctly_classified_only:
        correct_idxs, _, _ = graphs_correctly_classified(dataset_name)
        idxs = correct_idxs
    else:
        idxs = list(map(int, predictions.keys()))

    # build dictionary of idx → label, potentially filtered for correct classifications
    labels = {int(i): entry["true_label"] for i, entry in predictions.items() if int(i) in idxs}

    # generate pairs with same label
    pairs = set()
    for i, j in itertools.combinations(sorted(labels.keys()), 2):
        if labels[i] == labels[j]:
            pairs.add((i, j))

    # optionally, save result to file
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(sorted(list(pairs)), f, indent=2)
        print(f"Saved {len(pairs)} graph pairs to {save_path}")

    return pairs


def graph_index_pairs_diff_class(dataset_name,
                                 correctly_classified_only=True,
                                 save_path=None):
    """
    Returns all index pairs (i, j) from MUTAG where the graphs are of different class.
    If correctly_classified_only is True, only include graphs correctly classified by the model.
    Optionally saves the result to a JSON file.
    """
    # read in predictions of our model on MUTAG graphs
    with open(f"data/{dataset_name}/predictions/{dataset_name}_predictions.json") as f:
        predictions = json.load(f)

    # filter graph indexes, if only correctly classified graphs should be considered
    if correctly_classified_only:
        correct_idxs, _, _ = graphs_correctly_classified(dataset_name)
        idxs = correct_idxs
    else:
        idxs = list(map(int, predictions.keys()))

    # build dictionary of idx → label, potentially filtered for correct classifications
    labels = {int(i): entry["true_label"] for i, entry in predictions.items() if int(i) in idxs}

    # generate pairs with same label
    pairs = set()
    for i, j in itertools.combinations(sorted(labels.keys()), 2):
        if labels[i] != labels[j]:
            pairs.add((i, j))

    # optionally, save result to file
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(sorted(list(pairs)), f, indent=2)
        print(f"Saved {len(pairs)} graph pairs to {save_path}")

    return pairs