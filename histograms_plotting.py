import os

from plotting_utils import histogram_file, collect_existing_hist_files, decile_file, plot_histograms_for_cuts
from config import DATASET_NAME

# define input, output paths
ANALYSIS_DIR = f"data/{DATASET_NAME}/analysis"
PLOT_DIR = os.path.join(ANALYSIS_DIR, "plot", "histograms")
os.makedirs(PLOT_DIR, exist_ok=True)

# index set cuts
CUTS = [
    "same_class_all",
    "same_class_0_all",
    "same_class_1_all",
    "diff_class_all",
    "train_train_same",
    "train_train_same_0",
    "train_train_same_1",
    "train_train_diff",
    "test_test_same",
    "test_test_same_0",
    "test_test_same_1",
    "test_test_diff",
    "train_test_same",
    "train_test_same_0",
    "train_test_same_1",
    "train_test_diff",
]

# create index set to filename mappings
hist_map = {k: histogram_file(k) for k in CUTS}
decile_map = {k: decile_file(k) for k in CUTS}

if __name__ == "__main__":

    # create label to file path map
    label_to_file = collect_existing_hist_files(CUTS)
    if not label_to_file:
        raise SystemExit("No histogram files found — did you run count_paths_by_num_flips?")

    # ------------------------- histograms for same, same_0, same_1 over all splits ------------------------
    # absolute counts
    compare = ["same_class_all", "same_class_0_all, same_class_1_all"]
    plot_histograms_for_cuts(
        key_to_file=hist_map,
        cuts=compare,
        normalize=False,
        title=f"{DATASET_NAME}: flips-per-path (counts)",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_hist_counts_{'_'.join(compare)}.png"),
    )
    # normalized
    plot_histograms_for_cuts(
        key_to_file=hist_map,
        cuts=compare,
        normalize=True,
        title=f"{DATASET_NAME}: flips-per-path (normalized)",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_hist_norm_{'_'.join(compare)}.png"),
    )

    # --------------- histograms for same, train-train vs test-test -------------------------------------

    compare = ["train_train_same", "test_test_same", "train_test_same"]
    # absolute counts
    plot_histograms_for_cuts(
        key_to_file=hist_map,
        cuts=compare,
        normalize=False,
        title=f"{DATASET_NAME}: flips-per-path (counts)",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_hist_counts_{'_'.join(compare)}.png"),
    )
    # normalized
    plot_histograms_for_cuts(
        key_to_file=hist_map,
        cuts=compare,
        normalize=True,
        title=f"{DATASET_NAME}: flips-per-path (normalized)",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_hist_norm_{'_'.join(compare)}.png"),
    )

    # ------------------------- histograms diff class, train vs test -------------------------------------

    compare = ["train_train_diff", "test_test_diff", "train_test_diff"]
    # absolute counts
    plot_histograms_for_cuts(
        key_to_file=hist_map,
        cuts=compare,
        normalize=False,
        title=f"{DATASET_NAME}: flips-per-path (counts)",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_hist_counts_{'_'.join(compare)}.png"),
    )
    # normalized
    plot_histograms_for_cuts(
        key_to_file=hist_map,
        cuts=compare,
        normalize=True,
        title=f"{DATASET_NAME}: flips-per-path (normalized)",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_hist_norm_{'_'.join(compare)}.png"),
    )

