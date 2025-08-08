from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, Optional
import matplotlib.pyplot as plt


# ------------------ LOADERS ------------------

def load_histogram(path: str) -> Dict[int, int]:
    """
    Load a histogram {num_flips: count, ...} from JSON.
    Keys are stored as strings in JSON but converted to ints here.
    """
    with open(path, "r") as f:
        data = json.load(f)
    return {int(k): int(v) for k, v in data.items()}


def load_decile_distribution(path: str) -> Dict[str, float]:
    """
    Load a decile -> value mapping from JSON.
    Accepts keys as strings "0".."9". Values typically are proportions (sum ~1).
    """
    with open(path, "r") as f:
        d = json.load(f)

    if "avg_proportion" in d:
        d = d["avg_proportion"]

    return {str(k): float(v) for k, v in d.items() if str(k).isdigit()}


# ------------------ PLOTTING: NUM FLIPS PER PATH ------------------

def plot_num_flips_histograms_from_files(
    label_to_file: Dict[str, str],
    normalize: bool = False,
    title: str = "# Flips per Path",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Load precomputed num_flips histograms from JSON files and plot them together.

    Args:
        label_to_file: {"train_train": "...json", "test_test": "...json", ...}
        normalize: If True, bars show proportions.
        title: Figure title.
        save_path: If given, saves the figure there.
        show: If True, calls plt.show().
    """
    # load all histograms
    histograms = {label: load_histogram(path) for label, path in label_to_file.items()}

    all_k = sorted({k for h in histograms.values() for k in h.keys()})
    set_names = list(histograms.keys())
    n_sets = len(set_names)
    width = 0.8 / max(n_sets, 1)

    # normalize if requested
    processed = {}
    for name, h in histograms.items():
        if normalize:
            total = sum(h.values()) or 1
            processed[name] = {k: (v / total) for k, v in h.items()}
        else:
            processed[name] = dict(h)

    # plot
    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, name in enumerate(set_names):
        vals = [processed[name].get(k, 0) for k in all_k]
        ax.bar([k + idx*width for k in range(len(all_k))], vals, width=width, label=name)

    ax.set_xticks([k + (n_sets-1)*width/2 for k in range(len(all_k))])
    ax.set_xticklabels([str(k) for k in all_k])
    ax.set_xlabel("# flips")
    ax.set_ylabel("Proportion" if normalize else "Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    if save_path:
        Path(os.path.dirname(save_path) or ".").mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close(fig)


# ------------------ PLOTTING: RELATIVE DECILE DISTRIBUTIONS ------------------

def plot_decile_distributions_from_files(
    label_to_file: Dict[str, str],
    title: str = "Relative Flip Distribution by Decile",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Load precomputed relative decile distributions from JSON files and plot them.

    Args:
        label_to_file: {"same_class": "...json", "diff_class": "...json", ...}
        title: Figure title.
        save_path: If given, saves figure there.
        show: If True, calls plt.show().
    """
    distributions = {label: load_decile_distribution(path) for label, path in label_to_file.items()}

    xs = list(range(10))

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, decmap in distributions.items():
        ys = [float(decmap.get(str(k), 0.0)) for k in xs]
        ax.plot(xs, ys, marker="o", label=name)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{10*k}-{10*(k+1)}%" for k in xs])
    ax.set_xlabel("Edit-path decile (relative position)")
    ax.set_ylabel("Proportion of flips")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    if save_path:
        Path(os.path.dirname(save_path) or ".").mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close(fig)
