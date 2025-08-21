from datetime import datetime, timezone

from analyse_utils import *
from config import DATASET_NAME, CORRECTLY_CLASSIFIED_ONLY
from index_sets_utils import build_index_set_cuts

if __name__ == "__main__":

    # define output path
    out_path = f"data/{DATASET_NAME}/analysis/paths_per_num_flips/{DISTANCE_MODE}/" \
               f"{DATASET_NAME}_flip_distribution_per_num_flips_by_{DISTANCE_MODE}.json"

    # define inputs
    split_path = "model/best_split.json"
    MAX_K = 8

    # todo: this had an old input file. run again!!
    # retrieve data according to distance mode
    if DISTANCE_MODE == "cost":
        dist_path = f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_dist_per_path.json"
        flips_path = f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_flip_occurrences_per_path_by_cost.json"
    elif DISTANCE_MODE == "cost":
        dist_path = f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_num_ops_per_path.json"
        flips_path = f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_flip_occurrences_per_path_by_edit_step.json"
    else:
        print(f"[warn] config.DISTANCE_MODE has unexpected value {DISTANCE_MODE}. Expected 'cost' or 'num_ops."
              f"Assuming 'cost.")
        dist_path = f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_dist_per_path.json"
        flips_path = f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_flip_occurrences_per_path_by_cost.json"
    cuts = build_index_set_cuts(
        dataset_name=DATASET_NAME,
        correctly_classified_only=CORRECTLY_CLASSIFIED_ONLY,
        split_path=split_path,
    )

    keys = [
        "same_class_all", "same_class_0_all", "same_class_1_all", "diff_class_all",
        "same_train_train", "same_0_train_train", "same_1_train_train", "diff_train_train",
        "same_test_test", "same_0_test_test", "same_1_test_test", "diff_test_test",
        "same_train_test", "same_0_train_test", "same_1_train_test", "diff_train_test",
    ]

    # ---------------------- global ------------------------------

    print("→ Computing per-num-flips decile distribution (GLOBAL)")
    global_stats = flip_distribution_over_deciles_by_num_flips(
        max_num_flips=MAX_K,
        dist_input_path=dist_path,
        flips_input_path=flips_path,
        output_path=None,
        idx_pair_set=None,
    )

    # -------------------- per index set -------------------------
    per_index_set = {}
    for key in keys:
        idx_set = cuts[key]
        print(f"→ Computing per-num-flips decile distribution for {key} ({len(idx_set)} pairs)")
        stats = flip_distribution_over_deciles_by_num_flips(
            max_num_flips=MAX_K,
            dist_input_path=dist_path,
            flips_input_path=flips_path,
            output_path=None,
            idx_pair_set=idx_set,
        )
        per_index_set[key] = {
            "num_pairs": len(idx_set),
            **stats
        }

    # -------------------- save to file -------------------------
    data = {
        "meta": {
            "dataset": DATASET_NAME,
            "distance_mode": DISTANCE_MODE,
            "correctly_classified_only": CORRECTLY_CLASSIFIED_ONLY,
            "split_path": split_path,
            "dist_path": dist_path,
            "flips_path": flips_path,
            "max_num_flips": MAX_K,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "global": global_stats,
        "per_index_set": per_index_set
    }

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved combined per-num-flips decile distributions → {out_path}")
