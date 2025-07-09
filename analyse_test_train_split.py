import itertools
import json


if __name__ == "__main__":
    dataset_name = "MUTAG"

    with open("model/best_split.json") as f:
        split = json.load(f)

    with open(f"data/{dataset_name}/predictions/{dataset_name}_edit_path_predictions.json", "r") as f:
        predictions = json.load(f)

    with open(f"data/{dataset_name}/predictions/{dataset_name}_changes_per_path.json") as f:
        changes_dict = json.load(f)

    train_set = set(split["train_idx"])
    test_set = set(split["test_idx"])

    # working logic:

    stats = {
        'train_train': [],
        'test_test': [],
        'train_test': []
    }

    for pair_str, steps in changes_dict.items():
        i, j = map(int, pair_str.split(","))

        num_changes = len(steps)

        if i in train_set and j in train_set:
            stats['train_train'].append(num_changes)
        elif i in test_set and j in test_set:
            stats['test_test'].append(num_changes)
        else:
            stats['train_test'].append(num_changes)

    mean_changes = {'train_train': sum(stats['train_train'])/len(stats['train_train'])}
    print(stats)
    print(mean_changes)

    # -----------------------------------------------------------------------------
    ''' # alternative to workings below,
    # does not work yet due to change_dict having entries for (0,1), (0,2) only:

    stats_altern = {
        'train_train': [],
        'test_test': [],
        'train_test': []
    }

    for (i, j) in set(itertools.combinations(train_set, 2)):
        changes = changes_dict[f"{i},{j}"]
        num_changes = len(changes)
        stats_altern['train_train'].append(num_changes)

    for (i, j) in set(itertools.combinations(test_set, 2)):
        changes = changes_dict[f"{i},{j}"]
        num_changes = len(changes)
        stats_altern['test_test'].append(num_changes)

    for (i, j) in set(itertools.product(train_set, test_set)):
        changes = changes_dict[f"{i},{j}"]
        num_changes = len(changes)
        stats_altern['train_test'].append(num_changes) '''

