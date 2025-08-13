import os

from analyse_utils import flip_distribution_by_indexset
from config import DATASET_NAME, CORRECTLY_CLASSIFIED_ONLY, DISTANCE_MODE
from index_sets_utils import build_index_set_cuts

if __name__ == "__main__":

    # inputs
    split_path = "model/best_split.json"
    if DISTANCE_MODE == "cost_function":
        dist_path = f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_dist_per_pair.json"
    elif DISTANCE_MODE == "operations_count":
        dist_path = f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_num_ops_per_pair.json"
    else:
        print("Provide valid param for DISTANCE MODE in config.py.")
    flips_path = f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_changes_per_path.json"
    out_dir = f"data/{DATASET_NAME}/analysis"
    os.makedirs(out_dir, exist_ok=True)

    # build all cut index sets (same/diff class + train-train / test-test / train-test)
    cuts = build_index_set_cuts(
        dataset_name=DATASET_NAME,
        correctly_classified_only=CORRECTLY_CLASSIFIED_ONLY,
        split_path=split_path,
    )

    keys = [
        "same_class_all", "same_class_0_all", "same_class_1_all", "diff_class_all",
        "train_train_same", "train_train_same_0", "train_train_same_1", "train_train_diff",
        "test_test_same",  "test_test_same_0",  "test_test_same_1",  "test_test_diff",
        "train_test_same", "train_test_same_0", "train_test_same_1", "train_test_diff",
    ]

    for k in keys:
        pair_set = cuts[k]
        out_path = os.path.join(out_dir, f"{DATASET_NAME}_rel_flips_per_decile_{k}.json")
        print(f"Computing decile distribution for {k} ({len(pair_set)} pairs)")
        flip_distribution_by_indexset(
            idx_pair_set=pair_set,
            dist_input_path=dist_path,
            flips_input_path=flips_path,
            output_path=out_path,
        )
