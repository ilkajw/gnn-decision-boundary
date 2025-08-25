from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, Optional, Iterable, List
import matplotlib.pyplot as plt

from config import DATASET_NAME, DISTANCE_MODE


# ------------------- old helpers -> check deletion ----------------------------------

def collect_existing_hist_files(cuts) -> Dict[str, str]:
    label_to_file: Dict[str, str] = {}
    for key in cuts:
        path = histogram_file(key)
        if os.path.exists(path):
            label_to_file[key] = path
        else:
            print(f"[warn] Skipping missing histogram: {path}")
    return label_to_file


def histogram_file(key) -> str:
        return os.path.join(f"data/{DATASET_NAME}/analysis/num_flip_histogram/by_{DISTANCE_MODE}/"
                            f"{DATASET_NAME}_flips_hist_abs_by_{DISTANCE_MODE}_{key}.json")


def decile_file(key: str) -> str:
    return os.path.join(f"data/{DATASET_NAME}/analysis/decile_distribution",
                        f"{DATASET_NAME}_decile_distribution_{key}.json")


def load_histogram(path: str) -> Dict[int, int]:
    """
    Load a histogram {num_flips: count, ...} from JSON.
    Keys are stored as strings in JSON but converted to ints here.
    """
    with open(path, "r") as f:
        data = json.load(f)
    return {int(k): int(v) for k, v in data.items()}


def load_distribution_per_indexset(path: str, field: str = "avg_per_path", normalize_abs: bool = True):
    with open(path, "r") as f:
        d = json.load(f)
    rec = d[field]
    if field == "abs_counts" and normalize_abs:
        total = sum(rec.values()) or 1
        rec = {str(dd): rec.get(str(dd), 0) / total for dd in map(str, range(10))}
    return {str(dd): float(rec.get(str(dd), 0.0)) for dd in range(10)}


def plot_histograms_for_cuts(
    key_to_file: Dict[str, str],
    cuts: List[str],
    normalize: bool = False,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Plot #flips-per-path histograms for a small selection of cuts (e.g., up to 3).
    """
    # filter + check
    selected = {k: key_to_file[k] for k in cuts if k in key_to_file and os.path.exists(key_to_file[k])}
    if not selected:
        raise ValueError("None of the requested cuts have an existing histogram file.")
    if len(selected) > 3:
        print(f"⚠️ You selected {len(selected)} cuts; consider ≤3 for readability.")

    # load
    histograms = {label: load_histogram(path) for label, path in selected.items()}
    all_k = sorted({k for h in histograms.values() for k in h.keys()})
    names = list(histograms.keys())
    n = len(names)
    width = 0.8 / max(n, 1)

    # normalize if requested
    proc = {}
    for name, h in histograms.items():
        if normalize:
            total = sum(h.values()) or 1
            proc[name] = {k: (v / total) for k, v in h.items()}
        else:
            proc[name] = dict(h)

    # def color palette
    colors = ["blue", "orange", "yellow"]

    # plot
    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, name in enumerate(names):
        vals = [proc[name].get(k, 0) for k in all_k]
        xpos = [x + idx * width for x in range(len(all_k))]
        bars = ax.bar(
            xpos,
            vals,
            width=width,
            label=name,
            color=colors[idx % len(colors)],  # apply palette per series
        )
        # annotate values on top of bars
        for rect, y in zip(bars, vals):
            if normalize:
                label = f"{y:.2f}"
            else:
                label = f"{int(y)}"
            ax.annotate(
                label,
                xy=(rect.get_x() + rect.get_width() / 2, y),
                xytext=(0, 3),  # offset in points
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks([x + (n-1)*width/2 for x in range(len(all_k))])
    ax.set_xticklabels([str(k) for k in all_k])
    ax.set_xlabel("Anzahl Flips")
    ax.set_ylabel("Anteil Pfade" if normalize else "Anzahl Pfade")
    # ax.set_title(title or "Flips-per-path histogram")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    if save_path:
        Path(os.path.dirname(save_path) or ".").mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_deciles_for_cuts(
    key_to_file: Dict[str, str],
    cuts: List[str],
    field: str = "global_proportion",   # or "avg_per_path" / "abs_counts"
    normalize_abs: bool = False,         # only used if field == "abs_counts"
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Plot decile distributions for a small selection of cuts (e.g., up to 3) as a grouped histogram
    with value annotations on/above each bar.
    """
    # filter available files
    selected = {k: key_to_file[k] for k in cuts if k in key_to_file and os.path.exists(key_to_file[k])}
    if not selected:
        raise ValueError("None of the requested cuts have an existing decile file.")
    if len(selected) > 3:
        print(f"⚠️ You selected {len(selected)} cuts. Consider ≤3 for readability.")
    if len(selected) < len(cuts):
        print(f"⚠️ Cuts is: {cuts}. Selected is: {selected.keys()}.")

    xs = list(range(10))  # deciles 0..9
    names = list(selected.keys())
    n = len(names)
    width = 0.8 / max(n, 1)  # total cluster width ~0.8

    # load data
    series = []
    for name in names:
        decmap = load_distribution_per_indexset(
            selected[name], field=field, normalize_abs=normalize_abs
        )
        ys = [decmap[str(d)] for d in xs]
        series.append((name, ys))

    # plot grouped bars
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, (name, ys) in enumerate(series):
        xpos = [x + idx * width for x in xs]
        bars = ax.bar(xpos, ys, width=width, label=name)

        # annotate values on/above bars
        for rect, y in zip(bars, ys):
            if field == "abs_counts" and not normalize_abs:
                label = f"{int(round(y))}"
            else:
                # proportions or normalized counts
                label = f"{y:.2f}"
            height = rect.get_height()
            ax.annotate(
                label,
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # offset in points
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # nice ticks & labels
    ax.set_xticks([x + (n - 1) * width / 2 for x in xs])
    ax.set_xticklabels([f"{10*d}-{10*(d+1)}%" for d in xs])
    ax.set_xlabel("Dezil Edit-Pfad")
    ax.set_ylabel("Anteil Pfade" if field != "abs_counts" or normalize_abs else "Anzahl Pfade")
    # ax.set_title(title or f"Decile distribution ({field})")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    if save_path:
        Path(os.path.dirname(save_path) or ".").mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close(fig)
