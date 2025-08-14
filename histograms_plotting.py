import os

from plotting_utils import load_histograms, plot_histograms_from_dict
from config import DATASET_NAME, DISTANCE_MODE

# define input, output paths
ANALYSIS_DIR = f"data/{DATASET_NAME}/analysis"
PLOT_DIR = os.path.join(ANALYSIS_DIR, "plots", "histograms", DISTANCE_MODE)
os.makedirs(PLOT_DIR, exist_ok=True)

if __name__ == "__main__":

    # ---------------------- histograms for same, same_0, same_1 over all splits ------------------------

    compare = ["same_class_all", "same_class_0_all", "same_class_1_all"]

    # absolute counts
    h_abs = load_histograms(compare, normalize=False)
    if not h_abs:
        raise SystemExit("No histograms found in consolidated file — did you run the single-writer script?")
    plot_histograms_from_dict(
        histograms=h_abs,
        normalize=False,
        title=f"{DATASET_NAME}: flips-per-path (absolute, by {DISTANCE_MODE})",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_flips_hist_abs_by_{DISTANCE_MODE}_{'_'.join(compare)}.png"),
    )

    # normalized
    h_rel = load_histograms(compare, normalize=True)
    plot_histograms_from_dict(
        histograms=h_rel,
        normalize=True,
        title=f"{DATASET_NAME}: flips-per-path (normalized, by {DISTANCE_MODE})",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_flips_hist_norm_by_{DISTANCE_MODE}_{'_'.join(compare)}.png"),
    )

    # --------------- histograms for same, train-train vs test-test -------------------------------------

    compare = ["same_train_train", "same_test_test", "same_train_test"]

    h_abs = load_histograms(compare, normalize=False)
    plot_histograms_from_dict(
        histograms=h_abs,
        normalize=False,
        title=f"{DATASET_NAME}: flips-per-path (absolute, by {DISTANCE_MODE})",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_flips_hist_abs_by_{DISTANCE_MODE}_{'_'.join(compare)}.png"),
    )

    h_rel = load_histograms(compare, normalize=True)
    plot_histograms_from_dict(
        histograms=h_rel,
        normalize=True,
        title=f"{DATASET_NAME}: flips-per-path (normalized, by {DISTANCE_MODE})",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_flips_hist_norm_by_{DISTANCE_MODE}_{'_'.join(compare)}.png"),
    )

    # ------------------------- histograms diff class, train vs test -------------------------------------

    compare = ["diff_train_train", "diff_test_test", "diff_train_test"]

    h_abs = load_histograms(compare, normalize=False)
    plot_histograms_from_dict(
        histograms=h_abs,
        normalize=False,
        title=f"{DATASET_NAME}: flips-per-path (absolute, by {DISTANCE_MODE})",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_flips_hist_abs_by_{DISTANCE_MODE}_{'_'.join(compare)}.png"),
    )

    h_rel = load_histograms(compare, normalize=True)
    plot_histograms_from_dict(
        histograms=h_rel,
        normalize=True,
        title=f"{DATASET_NAME}: flips-per-path (normalized, by {DISTANCE_MODE})",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_flips_hist_norm_by_{DISTANCE_MODE}_{'_'.join(compare)}.png"),
    )
