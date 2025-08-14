import os
import json

from analyse_utils import count_paths_by_num_flips
from config import DATASET_NAME, CORRECTLY_CLASSIFIED_ONLY, DISTANCE_MODE
from index_sets_utils import build_index_set_cuts


def to_relative(counts_dict):
    # counts_dict: {"0": int, "1": int, ...} or {0: int, 1: int, ...}
    total = sum(counts_dict.values())
    if total == 0:
        return {str(k): 0.0 for k in counts_dict.keys()}
    return {str(k): (counts_dict[k] / total) for k in counts_dict.keys()}


if __name__ == "__main__":

    # inputs
    split_path = "model/best_split.json"

    if DISTANCE_MODE == "cost":
        flips_path = f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_flip_occurrences_per_path_by_cost.json"
    else:
        flips_path = f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_flip_occurrences_per_path_by_edit_step.json"

    out_dir = f"data/{DATASET_NAME}/analysis/num_flip_histogram/by_{DISTANCE_MODE}"
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
        ("same_train_train",   True),
        ("same_0_train_train", True),
        ("same_1_train_train", True),
        ("diff_train_train",   False),

        # test–test
        ("same_test_test",   True),
        ("same_0_test_test", True),
        ("same_1_test_test", True),
        ("diff_test_test",   False),

        # train–test
        ("same_train_test",   True),
        ("same_0_train_test", True),
        ("same_1_train_test", True),
        ("diff_train_test",   False),
    ]

    # run histograms & save individual + combined

    combined = {}
    combined_rel = {}

    for key, same_flag in keys_and_flags:
        idx_pair_set = cuts[key]
        out_path = os.path.join(out_dir, f"{DATASET_NAME}_flips_hist_abs_by_{DISTANCE_MODE}_{key}.json")
        print(f"→ counting flips histogram for {key} ({len(idx_pair_set)} pairs)")
        hist = count_paths_by_num_flips(
            idx_pair_set=idx_pair_set,
            flips_input_path=flips_path,
            output_path=out_path,
            same_class=same_flag,
        )
        combined[key] = hist

        # compute relative proportions and write to another file
        rel = to_relative(hist)
        rel_out_path = os.path.join(out_dir, f"{DATASET_NAME}_flips_hist_norm_by_{DISTANCE_MODE}_{key}.json")
        with open(rel_out_path, "w") as f:
            json.dump(rel, f, indent=2)

        # keep for combined relative output
        combined_rel[key] = rel

    # save merged histogram
    combined_path = os.path.join(out_dir, f"{DATASET_NAME}_flips_hist_abs_by_{DISTANCE_MODE}_all.json")
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Saved combined histograms → {combined_path}")

    # save merged (relative)
    combined_rel_path = os.path.join(out_dir, f"{DATASET_NAME}_flips_hist_norm_by_{DISTANCE_MODE}_all.json")
    with open(combined_rel_path, "w") as f:
        json.dump(combined_rel, f, indent=2)
    print(f"Saved combined (relative) → {combined_rel_path}")

