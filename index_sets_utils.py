import itertools
import json
import os

from config import DATASET_NAME


def load_split_sets(split_path):
    """Return train/test index sets from best_split.json."""
    with open(split_path, "r") as f:
        split = json.load(f)
    train = set(split["train_idx"])
    test = set(split["test_idx"])
    return train, test


def pairs_within(idx_set: set[int]) -> set[tuple[int, int]]:
    """All unordered pairs within one set."""
    return {tuple(sorted(p)) for p in itertools.combinations(idx_set, 2)}


def pairs_across(a_set: set[int], b_set: set[int]) -> set[tuple[int, int]]:
    """All unordered pairs across two disjoint sets."""
    return {tuple(sorted((i, j))) for i in a_set for j in b_set if i != j}


def cut_pairs(base_pairs, allowed_pairs):
    """Intersection: keep only pairs that are both in base_pairs and allowed_pairs."""
    return base_pairs & allowed_pairs


def build_index_set_cuts(dataset_name: str, correctly_classified_only: bool, split_path: str) -> dict[str, set[tuple[int,int]]]:
    """
    Returns a dict of pair-sets covering:
      - same_class_all / same_class_0_all / same_class_1_all / diff_class_all
      - train_train_same / train_train_diff / test_test_same / test_test_diff / train_test_same / train_test_diff
      - same-class (0/1) variants for each split bucket
    """
    # label-based global sets
    diff_class_pairs = graph_index_pairs_diff_class(
        dataset_name=dataset_name,
        correctly_classified_only=correctly_classified_only,
        save_path=f"data/{dataset_name}/index_sets/{dataset_name}_idx_pairs_diff_class.json",
    )
    same_class_pairs, same_class_0_pairs, same_class_1_pairs = graph_index_pairs_same_class(
        dataset_name=dataset_name,
        correctly_classified_only=correctly_classified_only,
        save_dir=f"data/{dataset_name}/index_sets/{dataset_name}_idx_pairs",
    )

    # train/test ids
    train_set, test_set = load_split_sets(split_path)

    # structural buckets
    tt_pairs = pairs_within(train_set)              # train–train
    uu_pairs = pairs_within(test_set)               # test–test
    tu_pairs = pairs_across(train_set, test_set)    # train–test

    # cuts between same/diff and test-test/train-train/test-train
    cuts = {
        # global label sets
        "same_class_all":   same_class_pairs,
        "same_class_0_all": same_class_0_pairs,
        "same_class_1_all": same_class_1_pairs,
        "diff_class_all":   diff_class_pairs,

        # train–train
        "train_train_same":  cut_pairs(tt_pairs, same_class_pairs),
        "train_train_diff":  cut_pairs(tt_pairs, diff_class_pairs),
        "train_train_same_0": cut_pairs(tt_pairs, same_class_0_pairs),
        "train_train_same_1": cut_pairs(tt_pairs, same_class_1_pairs),

        # test–test
        "test_test_same":   cut_pairs(uu_pairs, same_class_pairs),
        "test_test_diff":   cut_pairs(uu_pairs, diff_class_pairs),
        "test_test_same_0": cut_pairs(uu_pairs, same_class_0_pairs),
        "test_test_same_1": cut_pairs(uu_pairs, same_class_1_pairs),

        # train–test
        "train_test_same":   cut_pairs(tu_pairs, same_class_pairs),
        "train_test_diff":   cut_pairs(tu_pairs, diff_class_pairs),
        "train_test_same_0": cut_pairs(tu_pairs, same_class_0_pairs),
        "train_test_same_1": cut_pairs(tu_pairs, same_class_1_pairs),
    }

    return cuts


def graphs_correctly_classified(dataset_name):
    """Returns the indices of all graphs classified correctly by GAT model."""
    with open(f"data/{dataset_name}/predictions/{dataset_name}_predictions.json") as f:
        predictions = json.load(f)
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
