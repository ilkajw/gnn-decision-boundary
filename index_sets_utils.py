import itertools
import json
import os

from config import DATASET_NAME, CORRECTLY_CLASSIFIED_ONLY, PREDICTIONS_DIR, MODEL, MODEL_DIR, ROOT


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


def build_index_set_cuts(dataset_name=f"{DATASET_NAME}",
                         correctly_classified_only=CORRECTLY_CLASSIFIED_ONLY,
                         split_path=f"{MODEL_DIR}/{MODEL}_best_split.json",
                         ):
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
        save_path=f"{ROOT}/{DATASET_NAME}/index_sets/{dataset_name}_idx_pairs_diff_class.json",
    )
    same_class_pairs, same_class_0_pairs, same_class_1_pairs = graph_index_pairs_same_class(
        dataset=dataset_name,
        correctly_classified_only=correctly_classified_only,
        save_dir=f"{ROOT}/{DATASET_NAME}/index_sets/{dataset_name}_idx_pairs",
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
        "same_train_train":  cut_pairs(tt_pairs, same_class_pairs),
        "diff_train_train":  cut_pairs(tt_pairs, diff_class_pairs),
        "same_0_train_train": cut_pairs(tt_pairs, same_class_0_pairs),
        "same_1_train_train": cut_pairs(tt_pairs, same_class_1_pairs),

        # test–test
        "same_test_test":   cut_pairs(uu_pairs, same_class_pairs),
        "diff_test_test":   cut_pairs(uu_pairs, diff_class_pairs),
        "same_0_test_test": cut_pairs(uu_pairs, same_class_0_pairs),
        "same_1_test_test": cut_pairs(uu_pairs, same_class_1_pairs),

        # train–test
        "same_train_test":   cut_pairs(tu_pairs, same_class_pairs),
        "diff_train_test":   cut_pairs(tu_pairs, diff_class_pairs),
        "same_0_train_test": cut_pairs(tu_pairs, same_class_0_pairs),
        "same_1_train_test": cut_pairs(tu_pairs, same_class_1_pairs),
    }

    return cuts


def graphs_correctly_classified(dataset_name=DATASET_NAME):
    """Returns the indices of all original graphs classified correctly by model selected in config."""
    with open(f"{PREDICTIONS_DIR}/{dataset_name}_{MODEL}_predictions.json") as f:
        predictions = json.load(f)
    correct_idxs = [int(i) for i, entry in predictions.items() if entry["correct"]]
    return correct_idxs


# todo: potentially merge next two functions into 1 with same/diff argument
def graph_index_pairs_same_class(dataset=f"{DATASET_NAME}",
                                 correctly_classified_only=True,
                                 save_dir=None
                                 ):
    """
    Returns all index pairs (i, j) from original dataset where both graphs are of the same class.
    If correctly_classified_only is True, only include graphs correctly classified by the model.
    Optionally saves the result to a JSON file.

    Args:
        :param dataset: Name of graph dataset.
        :param correctly_classified_only: If True, only pairs of indices from correctly classified graphs
        will be considered.
        :param save_dir: File path where to save index pair list.
    """

    # read in predictions of our model on graphs in original dataset
    with open(f"{PREDICTIONS_DIR}/{dataset}_{MODEL}_predictions.json") as f:
        predictions = json.load(f)

    # optionally, filter for correctly classified graphs only
    if correctly_classified_only:
        idxs = graphs_correctly_classified(dataset)
    else:
        idxs = list(map(int, predictions.keys()))

    # map graph idx → true label
    labels = {int(i): entry["true_label"] for i, entry in predictions.items() if int(i) in idxs}

    # build sets of graph pairs
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

    # optionally, save all three same-class sets
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        def save_set(pair_set, suffix):
            file_path = os.path.join(save_dir, f"{dataset}_idx_pairs_{suffix}.json")
            with open(file_path, "w") as f:
                json.dump(sorted(list(pair_set)), f, indent=2)
            print(f"Saved {len(pair_set)} graph pairs to {file_path}")

        save_set(same_class_pairs, "same_class")
        save_set(same_class_0_pairs, "same_class_0")
        save_set(same_class_1_pairs, "same_class_1")

    return same_class_pairs, same_class_0_pairs, same_class_1_pairs


def graph_index_pairs_diff_class(dataset_name=DATASET_NAME,
                                 correctly_classified_only=True,
                                 save_path=None):
    """
    Returns all index pairs (i, j) from original datatset where the graphs are of different classes.
    If correctly_classified_only is True, only graphs correctly classified by the model are included.
    Optionally saves the result to a JSON file.

    Args:
        :param dataset_name: Name of graph dataset.
        :param correctly_classified_only: If True, only pairs of indices from correctly classified graphs
        will be considered.
        :param save_path: File path where to save index pair list.
    """

    # read in predictions of our model on original dataset graphs
    with open(f"{PREDICTIONS_DIR}/{dataset_name}_{MODEL}_predictions.json") as f:
        predictions = json.load(f)

    # optionally, filter for correctly classified graphs only
    if correctly_classified_only:
        correct_idxs = graphs_correctly_classified(dataset_name)
        idxs = correct_idxs
    else:
        idxs = list(map(int, predictions.keys()))

    # build dictionary of graph idx → true label
    labels = {int(i): entry["true_label"] for i, entry in predictions.items() if int(i) in idxs}

    # generate set of all graph pairs
    pairs = set()
    for i, j in itertools.combinations(sorted(labels.keys()), 2):
        if labels[i] != labels[j]:
            pairs.add((i, j))

    # optionally, save set to file
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(sorted(list(pairs)), f, indent=2)
        print(f"Saved {len(pairs)} graph pairs to {save_path}")

    return pairs
