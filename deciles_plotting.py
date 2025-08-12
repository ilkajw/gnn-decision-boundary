import os

from plotting_utils import histogram_file, collect_existing_hist_files, decile_file, plot_deciles_for_cuts
from config import DATASET_NAME

# define input, output paths
ANALYSIS_DIR = f"data/{DATASET_NAME}/analysis"
PLOT_DIR = os.path.join(ANALYSIS_DIR, "plot", "deciles")
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

    # ---------------------- same vs. diff ----------------------------------------

    compare = ["same_class_all", "diff_class_all"]
    # global proportion
    plot_deciles_for_cuts(
        key_to_file=decile_map,
        cuts=compare,
        field="global_proportion",
        normalize_abs=True,
        title=f"{DATASET_NAME}: deciles (global proportion)",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_deciles_global_{'_'.join(compare)}.png"),
    )
    # avg per-path
    plot_deciles_for_cuts(
        key_to_file=decile_map,
        cuts=compare,
        field="avg_per_path",
        normalize_abs=True,
        title=f"{DATASET_NAME}: deciles (average per-path)",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_avg_path_{'_'.join(compare)}.png"),
    )

    plot_deciles_for_cuts(
        key_to_file=decile_map,
        cuts=compare,
        field="abs_counts",
        normalize_abs=False,
        title=f"{DATASET_NAME}: deciles (absolute counts)",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_abs_counts_{'_'.join(compare)}.png"),
    )

    # ----------------- same vs same_0 vs. same_1 --------------------------------

    compare = ["same_class_all", "same_class_0_all", "same_class_1_all"]
    # global proportion
    plot_deciles_for_cuts(
        key_to_file=decile_map,
        cuts=compare,
        field="global_proportion",
        normalize_abs=True,
        title=f"{DATASET_NAME}: deciles (global proportion)",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_deciles_global_{'_'.join(compare)}.png"),
    )
    # average per-path
    plot_deciles_for_cuts(
        key_to_file=decile_map,
        cuts=compare,
        field="avg_per_path",
        normalize_abs=True,
        title=f"{DATASET_NAME}: deciles (average per-path)",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_avg_path_{'_'.join(compare)}.png"),
    )
    # absolute counts
    plot_deciles_for_cuts(
        key_to_file=decile_map,
        cuts=compare,
        field="abs_counts",
        normalize_abs=False,
        title=f"{DATASET_NAME}: deciles (absolute counts)",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_abs_counts_{'_'.join(compare)}.png"),
    )
