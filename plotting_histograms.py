import json
import os

from pathlib import Path
from typing import Dict, Optional, List
from matplotlib import pyplot as plt

from config import DATASET_NAME, DISTANCE_MODE

# ---------- define input, output paths -------------

ANALYSIS_DIR = f"data_control/{DATASET_NAME}/analysis"
PLOT_DIR = os.path.join(ANALYSIS_DIR, "plots", "histograms", f"by_{DISTANCE_MODE}")
os.makedirs(PLOT_DIR, exist_ok=True)


# ----------------- helpers -------------------

def histograms_file() -> str:
    return os.path.join(ANALYSIS_DIR,
                        f"flip_histograms/by_{DISTANCE_MODE}/{DATASET_NAME}_flips_hist_by_{DISTANCE_MODE}.json"
                        )


def load_histograms(keys: List[str], normalized: bool) -> Dict[str, Dict[int, float]]:
    """
    Load {key -> {num_flips: value}} from the single consolidated JSON.
    If normalized=False, pulls 'hist_abs'.
    If normalized=True,  pulls 'hist_rel'.
    Converts all flip-count keys to ints.
    """
    path = histograms_file()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Consolidated histogram file not found: {path}\n"
            "Did you run the script which writes the single .json?"
        )

    with open(path, "r") as f:
        data = json.load(f)

    results = data.get("results", {})
    field = "hist_rel" if normalized else "hist_abs"

    out: Dict[str, Dict[int, float]] = {}
    for k in keys:
        if k not in results:
            print(f"[warn] Missing key in consolidated results: {k}")
            continue
        hist = results[k].get(field, {})
        # keys may be strings or ints, normalize to ints
        out[k] = {int(kk): float(v) for kk, v in hist.items()}

    return out


def totals_from_abs(h_abs: Dict[str, Dict[int, float]]) -> Dict[str, int]:
    """
    Given absolute histograms {series -> {segment -> count}},
    return {series -> total_count}.
    """
    return {name: int(round(sum(bins.values()))) for name, bins in h_abs.items()}


def plot_histograms_from_dict(
        histograms: Dict[str, Dict[int, float]],
        totals: Optional[Dict[str, int]] = None,
        normalized: bool = False,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = False,
):
    """
    Plot #flips-per-path histograms from an in-memory dict
    {label -> {num_flips: value}}, where value is a count (if not normalized)
    or a proportion (if normalized).
    """
    if not histograms:
        raise ValueError("No histograms provided.")

    all_k = sorted({k for h in histograms.values() for k in h.keys()})
    names = list(histograms.keys())
    n_series = len(names)
    width = 0.8 / max(n_series, 1)

    # fixed color palette (blue, grey, yellow)
    colors = ["#1f77b4", '#808080', "#f2c94c"]

    fig, ax = plt.subplots(figsize=(9, 5))

    for idx, name in enumerate(names):
        vals = [histograms[name].get(k, 0.0) for k in all_k]
        xpos = [x + idx * width for x in range(len(all_k))]

        # insert number of contributing paths to legend if available
        if totals is not None and name in totals:
            label_text = f"{name} (n={int(totals[name])})"
        else:
            print(f"[warn] totals not available for '{name}'.")
            label_text = name

        # define plot bar per series
        bars = ax.bar(
            xpos,
            vals,
            width=width,
            label=label_text,
            color=colors[idx % len(colors)],
            linewidth=0.5,
        )

        # annotate values on bars
        for rect, y in zip(bars, vals):
            label = f"{y: .2f}" if normalized else f"{int(y)}"
            ax.annotate(
                label,
                xy=(rect.get_x() + rect.get_width() / 2, y),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks([x + (n_series - 1) * width / 2 for x in range(len(all_k))])
    ax.set_xticklabels([str(k) for k in all_k])
    ax.set_xlabel("Anzahl Flips")
    ax.set_ylabel("Anteil Pfade" if normalized else "Anzahl Pfade")
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    if save_path:
        Path(os.path.dirname(save_path) or ".").mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close(fig)


# --------------- run plotting ----------------

if __name__ == "__main__":

    # -------- histograms for same, same_0, same_1 ------------

    idx_sets = ["same_class_all", "same_class_0_all", "same_class_1_all"]

    # absolute values
    h_abs = load_histograms(idx_sets, normalized=False)
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
    h_rel = load_histograms(idx_sets, normalized=True)
    plot_histograms_from_dict(
        histograms=h_rel,
        totals=abs_totals,
        normalized=True,
        title=None,
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_flips_hist_norm_by_{DISTANCE_MODE}_{'_'.join(idx_sets)}.png"),
    )

    # -------------- histograms for same, train-train vs test-test -------------------

    idx_sets = ["same_train_train", "same_test_test", "same_train_test"]

    # absolute values
    h_abs = load_histograms(idx_sets, normalized=False)
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
    h_rel = load_histograms(idx_sets, normalized=True)
    plot_histograms_from_dict(
        histograms=h_rel,
        totals=abs_totals,
        normalized=True,
        title=None,
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_flips_hist_norm_by_{DISTANCE_MODE}_{'_'.join(idx_sets)}.png"),
    )

    # -------------- histograms diff class, train vs test ----------------

    idx_sets = ["diff_train_train", "diff_test_test", "diff_train_test"]

    # absolute values
    h_abs = load_histograms(idx_sets, normalized=False)
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
    h_rel = load_histograms(idx_sets, normalized=True)
    plot_histograms_from_dict(
        histograms=h_rel,
        totals=abs_totals,
        normalized=True,
        title=None,
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_flips_hist_norm_by_{DISTANCE_MODE}_{'_'.join(idx_sets)}.png"),
    )
