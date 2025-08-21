import os
from typing import Dict

from plotting_utils import load_histograms, plot_histograms_from_dict
from config import DATASET_NAME, DISTANCE_MODE

# define input, output paths
ANALYSIS_DIR = f"data/{DATASET_NAME}/analysis"
PLOT_DIR = os.path.join(ANALYSIS_DIR, "plots", "histograms", DISTANCE_MODE)
os.makedirs(PLOT_DIR, exist_ok=True)


# --------------- helper ----------------------------

def totals_from_abs(h_abs: Dict[str, Dict[int, float]]) -> Dict[str, int]:
    """
    Given absolute histograms {series -> {bin -> count}},
    return {series -> total_count}.
    """
    return {name: int(round(sum(bins.values()))) for name, bins in h_abs.items()}


# -------------------------- run plotting -------------------------------

if __name__ == "__main__":

    # ---------------------- histograms for same, same_0, same_1 over all splits ------------------------

    idx_sets = ["same_class_all", "same_class_0_all", "same_class_1_all"]

    # absolute values
    h_abs = load_histograms(idx_sets, normalize=False)
    if not h_abs:
        raise SystemExit("No histograms found in consolidated file — did you run the single-writer script?")
    abs_totals = totals_from_abs(h_abs)
    plot_histograms_from_dict(
        histograms=h_abs,
        totals=abs_totals,
        normalized=False,
        title=None,
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_flips_hist_abs_by_{DISTANCE_MODE}_{'_'.join(idx_sets)}.png"),
    )

    # normalized values
    h_rel = load_histograms(idx_sets, normalize=True)
    plot_histograms_from_dict(
        histograms=h_rel,
        totals=abs_totals,
        normalized=True,
        title=None,
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_flips_hist_norm_by_{DISTANCE_MODE}_{'_'.join(idx_sets)}.png"),
    )

    # --------------- histograms for same, train-train vs test-test -------------------------------------

    idx_sets = ["same_train_train", "same_test_test", "same_train_test"]

    # absolute values
    h_abs = load_histograms(idx_sets, normalize=False)
    if not h_abs:
        raise SystemExit("No histograms found in consolidated file — did you run the single-writer script?")
    abs_totals = totals_from_abs(h_abs)
    plot_histograms_from_dict(
        histograms=h_abs,
        totals=abs_totals,
        normalized=False,
        title=None,
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_flips_hist_abs_by_{DISTANCE_MODE}_{'_'.join(idx_sets)}.png"),
    )

    # normalized values
    h_rel = load_histograms(idx_sets, normalize=True)
    plot_histograms_from_dict(
        histograms=h_rel,
        totals=abs_totals,
        normalized=True,
        title=None,
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_flips_hist_norm_by_{DISTANCE_MODE}_{'_'.join(idx_sets)}.png"),
    )

    # ------------------------- histograms diff class, train vs test -------------------------------------

    idx_sets = ["diff_train_train", "diff_test_test", "diff_train_test"]

    # absolute values
    h_abs = load_histograms(idx_sets, normalize=False)
    if not h_abs:
        raise SystemExit("No histograms found in consolidated file — did you run the single-writer script?")
    abs_totals = totals_from_abs(h_abs)
    plot_histograms_from_dict(
        histograms=h_abs,
        totals=abs_totals,
        normalized=False,
        title=None,
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_flips_hist_abs_by_{DISTANCE_MODE}_{'_'.join(idx_sets)}.png"),
    )

    # normalized values
    h_rel = load_histograms(idx_sets, normalize=True)
    plot_histograms_from_dict(
        histograms=h_rel,
        totals=abs_totals,
        normalized=True,
        title=None,
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_flips_hist_norm_by_{DISTANCE_MODE}_{'_'.join(idx_sets)}.png"),
    )
