import json
from itertools import combinations

# todo: rewrite to dictionary "index: class".
#  possibly obsolete as logic changed this way: predict all graphs,
#  then add metadata on source/target classes, train/test split, correct classifications


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



