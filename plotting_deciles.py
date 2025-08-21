import os

from plotting_utils import load_deciles, plot_deciles_from_dict
from config import DATASET_NAME

# define input, output paths
ANALYSIS_DIR = f"data/{DATASET_NAME}/analysis"
PLOT_DIR = os.path.join(ANALYSIS_DIR, "plots", "decile_distribution")
os.makedirs(PLOT_DIR, exist_ok=True)

if __name__ == "__main__":

    # ----------------------- same vs. diff ----------------------------------------

    compare = ["same_class_all", "diff_class_all"]

    plot_deciles_from_dict(
        deciles=load_deciles(compare, field="global_proportion", normalize_abs=False),
        field="global_proportion",
        normalize_abs=False,
        title=f"{DATASET_NAME}: deciles (global proportion)",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_deciles_global_{'_'.join(compare)}.png"),
    )

    plot_deciles_from_dict(
        deciles=load_deciles(compare, field="avg_per_path", normalize_abs=False),
        field="avg_per_path",
        normalize_abs=False,
        title=f"{DATASET_NAME}: deciles (average per-path)",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_avg_path_{'_'.join(compare)}.png"),
    )

    plot_deciles_from_dict(
        deciles=load_deciles(compare, field="abs_counts", normalize_abs=False),
        field="abs_counts",
        normalize_abs=False,
        title=f"{DATASET_NAME}: deciles (absolute counts)",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_abs_counts_{'_'.join(compare)}.png"),
    )

    # ----------------- same vs same_0 vs. same_1 ----------------------------------

    compare = ["same_class_all", "same_class_0_all", "same_class_1_all"]

    plot_deciles_from_dict(
        deciles=load_deciles(compare, field="global_proportion", normalize_abs=False),
        field="global_proportion",
        normalize_abs=False,
        title=f"{DATASET_NAME}: deciles (global proportion)",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_deciles_global_{'_'.join(compare)}.png"),
    )

    plot_deciles_from_dict(
        deciles=load_deciles(compare, field="avg_per_path", normalize_abs=False),
        field="avg_per_path",
        normalize_abs=False,
        title=f"{DATASET_NAME}: deciles (average per-path)",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_avg_path_{'_'.join(compare)}.png"),
    )

    plot_deciles_from_dict(
        deciles=load_deciles(compare, field="abs_counts", normalize_abs=False),
        field="abs_counts",
        normalize_abs=False,
        title=f"{DATASET_NAME}: deciles (absolute counts)",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_abs_counts_{'_'.join(compare)}.png"),
    )

    # --------------- same: train–train vs test–test vs train–test ------------------

    compare = ["same_train_train", "same_test_test", "same_train_test"]

    plot_deciles_from_dict(
        deciles=load_deciles(compare, field="global_proportion", normalize_abs=False),
        field="global_proportion",
        normalize_abs=False,
        title=f"{DATASET_NAME}: deciles (global proportion)",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_deciles_global_{'_'.join(compare)}.png"),
    )

    plot_deciles_from_dict(
        deciles=load_deciles(compare, field="avg_per_path", normalize_abs=False),
        field="avg_per_path",
        normalize_abs=False,
        title=f"{DATASET_NAME}: deciles (average per-path)",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_avg_path_{'_'.join(compare)}.png"),
    )

    plot_deciles_from_dict(
        deciles=load_deciles(compare, field="abs_counts", normalize_abs=False),
        field="abs_counts",
        normalize_abs=False,
        title=f"{DATASET_NAME}: deciles (absolute counts)",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_abs_counts_{'_'.join(compare)}.png"),
    )

    # ----------------- diff: train–train vs test–test vs train–test ----------------

    compare = ["diff_train_train", "diff_test_test", "diff_train_test"]

    plot_deciles_from_dict(
        deciles=load_deciles(compare, field="global_proportion", normalize_abs=False),
        field="global_proportion",
        normalize_abs=False,
        title=f"{DATASET_NAME}: deciles (global proportion)",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_deciles_global_{'_'.join(compare)}.png"),
    )

    plot_deciles_from_dict(
        deciles=load_deciles(compare, field="avg_per_path", normalize_abs=False),
        field="avg_per_path",
        normalize_abs=False,
        title=f"{DATASET_NAME}: deciles (average per-path)",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_avg_path_{'_'.join(compare)}.png"),
    )

    plot_deciles_from_dict(
        deciles=load_deciles(compare, field="abs_counts", normalize_abs=False),
        field="abs_counts",
        normalize_abs=False,
        title=f"{DATASET_NAME}: deciles (absolute counts)",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_abs_counts_{'_'.join(compare)}.png"),
    )
