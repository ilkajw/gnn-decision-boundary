import json
import os
from datetime import datetime

from analyse_utils import flip_distribution_over_deciles_by_indexset
from config import DATASET_NAME, CORRECTLY_CLASSIFIED_ONLY, DISTANCE_MODE
from index_sets_utils import build_index_set_cuts

if __name__ == "__main__":

    # define output path
    out_dir = f"data/{DATASET_NAME}/analysis/decile_distribution/by_{DISTANCE_MODE}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        f"{DATASET_NAME}_flip_distribution_STATS_by_{DISTANCE_MODE}.json"
    )

    # define inputs
    split_path = "model/best_split.json"

    # retrieve data according to distance mode (cost vs. num_ops)
    if DISTANCE_MODE == "cost":
        dist_path = f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_dist_per_path.json"
        flips_path = f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_flip_occurrences_per_path_by_cost.json"
    else:
        dist_path = f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_num_ops_per_path.json"
        flips_path = f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_flip_occurrences_per_path_by_edit_step.json"

    # build all cut index sets (same/diff class + train-train / test-test / train-test)
    cuts = build_index_set_cuts(
        dataset_name=DATASET_NAME,
        correctly_classified_only=CORRECTLY_CLASSIFIED_ONLY,
        split_path=split_path,
    )

    keys = [
        "same_class_all", "same_class_0_all", "same_class_1_all", "diff_class_all",
        "same_train_train", "same_0_train_train", "same_1_train_train", "diff_train_train",
        "same_test_test",  "same_0_test_test",  "same_1_test_test",  "diff_test_test",
        "same_train_test", "same_0_train_test", "same_1_train_test", "diff_train_test",
    ]
    # todo: filter global by classified crrectly like sets
    # ------------------------------ global distribution --------------------------

    all_stats = flip_distribution_over_deciles_by_indexset(
        idx_pair_set=None,
        dist_input_path=dist_path,
        flips_input_path=flips_path
    )

    # ---------------------------- per index set distribution -----------------------

    per_set_stats = {}
    for key in keys:
        idx_set = cuts[key]
        print(f"Computing decile distribution for {key} ({len(idx_set)} pairs)")
        per_set_stats[key] = flip_distribution_over_deciles_by_indexset(
            idx_pair_set=idx_set,
            dist_input_path=dist_path,
            flips_input_path=flips_path,
            output_path=None,
        )

    # combine info on all index sets
    combined = {
        "meta": {
            "dataset": DATASET_NAME,
            "distance_mode": DISTANCE_MODE,
            "correctly_classified_only": CORRECTLY_CLASSIFIED_ONLY,
            "split_path": split_path,
            "dist_path": dist_path,
            "flips_path": flips_path,
            "generated_at": datetime.utcnow().isoformat() + "Z"
        },
        "global": all_stats,
        "per_index_set": per_set_stats,
    }

    # ----------------------- save ----------------------------------------

    out_path = os.path.join(
        out_dir,
        f"{DATASET_NAME}_flip_distribution_STATS_by_{DISTANCE_MODE}.json"
    )
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"Saved combined flip distribution stats to {out_path}")
