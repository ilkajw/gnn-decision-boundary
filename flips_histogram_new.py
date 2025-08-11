import os
import json

from analyse_utils import count_paths_by_num_flips
from config import DATASET_NAME, CORRECTLY_CLASSIFIED_ONLY
from index_sets_utils import build_index_set_cuts

if __name__ == "__main__":

    # inputs
    split_path = "model/best_split.json"  # adjust if needed
    flips_input = f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_changes_per_path.json"
    out_dir = f"data/{DATASET_NAME}/analysis"
    os.makedirs(out_dir, exist_ok=True)

    # build all index-set cuts (same/diff + train/train, test/test, train/test)
    cuts = build_index_set_cuts(
        dataset_name=DATASET_NAME,
        correctly_classified_only=CORRECTLY_CLASSIFIED_ONLY,
        split_path=split_path,
    )

    # key -> whether this cut is "same_class" (affects parity checks inside count_paths_by_num_flips)
    keys_and_flags = [
        # global by label
        ("same_class_all",   True),
        ("same_class_0_all", True),
        ("same_class_1_all", True),
        ("diff_class_all",   False),

        # train–train
        ("train_train_same",   True),
        ("train_train_same_0", True),
        ("train_train_same_1", True),
        ("train_train_diff",   False),

        # test–test
        ("test_test_same",   True),
        ("test_test_same_0", True),
        ("test_test_same_1", True),
        ("test_test_diff",   False),

        # train–test
        ("train_test_same",   True),
        ("train_test_same_0", True),
        ("train_test_same_1", True),
        ("train_test_diff",   False),
    ]

    # run histograms & save individual + combined
    combined = {}
    for key, same_flag in keys_and_flags:
        idx_pair_set = cuts[key]
        out_path = os.path.join(out_dir, f"{DATASET_NAME}_num_paths_per_num_flips_{key}.json")
        print(f"→ counting flips histogram for {key} ({len(idx_pair_set)} pairs)")
        hist = count_paths_by_num_flips(
            idx_pair_set=idx_pair_set,
            flips_input_path=flips_input,
            output_path=out_path,
            same_class=same_flag,
        )
        combined[key] = hist

    # save merged histogram
    combined_path = os.path.join(out_dir, f"{DATASET_NAME}_num_paths_per_num_flips_ALL_CUTS.json")
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Saved combined histograms → {combined_path}")
