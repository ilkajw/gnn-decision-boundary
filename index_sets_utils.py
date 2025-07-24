import itertools
import json
import os

from config import DATASET_NAME


def graphs_correctly_classified(dataset_name):

    """Returns the indices of all MUTAG graphs classified correctly by our GAT model."""

    with open(f"data/{dataset_name}/predictions/{dataset_name}_predictions.json") as f:
        predictions = json.load(f)
    # todo: not very intuitive for this to happen in this function.
    #  better add filtering into scripts
    correct_idxs = [int(i) for i, entry in predictions.items() if entry["correct"]]

    return correct_idxs


# todo: potentially merge next two functions into 1 with same/diff argument
def graph_index_pairs_same_class(dataset_name,
                                 correctly_classified_only=True,
                                 save_dir=None):
    """
    Returns all index pairs (i, j) from MUTAG where both graphs are of the same class.
    If correctly_classified_only is True, only include graphs correctly classified by the model.
    Optionally saves the result to a JSON file.

    Args:
        :param dataset_name: Name of graph dataset.
        :param correctly_classified_only: If True, only pairs of indices from correctly classified graphs
        will be considered.
        :param save_dir: File path where to save index pair list.
    """

    # read in predictions of our model on MUTAG graphs
    with open(f"data/{dataset_name}/predictions/{dataset_name}_predictions.json") as f:
        predictions = json.load(f)

    # filter graph indices
    if correctly_classified_only:
        idxs = graphs_correctly_classified(dataset_name)
    else:
        idxs = list(map(int, predictions.keys()))

    # map idx → label
    labels = {int(i): entry["true_label"] for i, entry in predictions.items() if int(i) in idxs}

    # prepare sets
    same_class_pairs = set()
    same_class_0_pairs = set()
    same_class_1_pairs = set()

    for i, j in itertools.combinations(sorted(labels.keys()), 2):
        if labels[i] == labels[j]:
            same_class_pairs.add((i, j))
            if labels[i] == 0:
                same_class_0_pairs.add((i, j))
            elif labels[i] == 1:
                same_class_1_pairs.add((i, j))

    # optionally save all three same-class sets
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        def save_set(pair_set, suffix):
            file_path = os.path.join(save_dir, f"{DATASET_NAME}_idx_pairs_{suffix}.json")
            with open(file_path, "w") as f:
                json.dump(sorted(list(pair_set)), f, indent=2)
            print(f"Saved {len(pair_set)} graph pairs to {file_path}")

        save_set(same_class_pairs, "same_class")
        save_set(same_class_0_pairs, "same_class_0")
        save_set(same_class_1_pairs, "same_class_1")

    return same_class_pairs, same_class_0_pairs, same_class_1_pairs


def graph_index_pairs_diff_class(dataset_name,
                                 correctly_classified_only=True,
                                 save_path=None):
    """
    Returns all index pairs (i, j) from MUTAG where the graphs are of different classes.
    If correctly_classified_only is True, only include graphs correctly classified by the model.
    Optionally saves the result to a JSON file.

    Args:
        :param dataset_name: Name of graph dataset.
        :param correctly_classified_only: If True, only pairs of indices from correctly classified graphs
        will be considered.
        :param save_path: File path where to save index pair list.
    """

    # read in predictions of our model on MUTAG graphs
    with open(f"data/{dataset_name}/predictions/{dataset_name}_predictions.json") as f:
        predictions = json.load(f)

    # filter graph indexes, if only correctly classified graphs should be considered
    if correctly_classified_only:
        correct_idxs = graphs_correctly_classified(dataset_name)
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
