import os

from plotting_utils import histogram_file, collect_existing_hist_files, decile_file, plot_histograms_for_cuts
from config import DATASET_NAME, DISTANCE_MODE

# define input, output paths
ANALYSIS_DIR = f"data/{DATASET_NAME}/analysis"
PLOT_DIR = os.path.join(ANALYSIS_DIR, "plots", "histograms", DISTANCE_MODE)
os.makedirs(PLOT_DIR, exist_ok=True)

# index set cuts
keys = [
        "same_class_all", "same_class_0_all", "same_class_1_all", "diff_class_all",
        "same_train_train", "same_0_train_train", "same_1_train_train", "diff_train_train",
        "same_test_test",  "same_0_test_test",  "same_1_test_test",  "diff_test_test",
        "same_train_test", "same_0_train_test", "same_1_train_test", "diff_train_test",
    ]

# create index set to filename mappings
hist_map = {k: histogram_file(k) for k in keys}
decile_map = {k: decile_file(k) for k in keys}

if __name__ == "__main__":

    # create label to file path map, paths are set according to 'DISTANCE_MODE'
    label_to_file = collect_existing_hist_files(keys)

    if not label_to_file:
        raise SystemExit("No histogram files found — did you run count_paths_by_num_flips?")

    print(label_to_file)

    # ------------------------- histograms for same, same_0, same_1 over all splits ------------------------

    compare = ["same_class_all", "same_class_0_all", "same_class_1_all"]
    # absolute counts
    plot_histograms_for_cuts(
        key_to_file=hist_map,
        cuts=compare,
        normalize=False,
        title=f"{DATASET_NAME}: flips-per-path (counts)",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_flips_hist_abs_by_{DISTANCE_MODE}_{'_'.join(compare)}.png")
    )
    # normalized
    plot_histograms_for_cuts(
        key_to_file=hist_map,
        cuts=compare,
        normalize=True,
        title=f"{DATASET_NAME}: flips-per-path (normalized)",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_flips_hist_norm_by_{DISTANCE_MODE}_{'_'.join(compare)}.png")
    )

    # --------------- histograms for same, train-train vs test-test -------------------------------------

    compare = ["same_train_train", "same_test_test", "same_train_test"]
    # absolute counts
    plot_histograms_for_cuts(
        key_to_file=hist_map,
        cuts=compare,
        normalize=False,
        title=f"{DATASET_NAME}: flips-per-path (counts)",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_flips_hist_abs_by_{DISTANCE_MODE}_{'_'.join(compare)}.png")
    )
    # normalized
    plot_histograms_for_cuts(
        key_to_file=hist_map,
        cuts=compare,
        normalize=True,
        title=f"{DATASET_NAME}: flips-per-path (normalized)",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_flips_hist_norm_by_{DISTANCE_MODE}_{'_'.join(compare)}.png")
    )

    # ------------------------- histograms diff class, train vs test -------------------------------------

    compare = ["diff_train_train", "diff_test_test", "diff_train_test"]
    # absolute counts
    plot_histograms_for_cuts(
        key_to_file=hist_map,
        cuts=compare,
        normalize=False,
        title=f"{DATASET_NAME}: flips-per-path (counts; by {DISTANCE_MODE})",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_flips_hist_abs_by_{DISTANCE_MODE}_{'_'.join(compare)}.png")
    )
    # normalized
    plot_histograms_for_cuts(
        key_to_file=hist_map,
        cuts=compare,
        normalize=True,
        title=f"{DATASET_NAME}: flips-per-path (normalized, by {DISTANCE_MODE})",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_flips_hist_norm_by_{DISTANCE_MODE}_{'_'.join(compare)}.png")
    )

