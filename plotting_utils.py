from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, Optional, Iterable, List
import matplotlib.pyplot as plt

from config import DATASET_NAME, DISTANCE_MODE


# ------------------ new helpers for histogram plotting ------------------------------

def histograms_file() -> str:
    return os.path.join(
        f"data/{DATASET_NAME}/analysis/flip_histograms/by_{DISTANCE_MODE}",
        f"{DATASET_NAME}_flips_hist_by_{DISTANCE_MODE}.json"
    )


def load_histograms(keys: List[str], normalize: bool) -> Dict[str, Dict[int, float]]:
    """
    Load {key -> {num_flips: value}} from the single consolidated JSON.
    If normalize=False, pulls 'hist_abs'.
    If normalize=True,  pulls 'hist_rel'.
    Converts all flip-count keys to ints.
    """
    path = histograms_file()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Consolidated histogram file not found: {path}\n"
            "Did you run the script that writes the single ALL.json?"
        )

    with open(path, "r") as f:
        data = json.load(f)

    results = data.get("results", {})
    field = "hist_rel" if normalize else "hist_abs"

    out: Dict[str, Dict[int, float]] = {}
    for k in keys:
        if k not in results:
            # keep a soft warning, skip silently otherwise
            print(f"[warn] Missing key in consolidated results: {k}")
            continue
        hist = results[k].get(field, {})
        # keys may be strings or ints; normalize to ints
        out[k] = {int(kk): float(v) for kk, v in hist.items()}

    return out


def plot_histograms_from_dict(
    histograms: Dict[str, Dict[int, float]],
    normalize: bool = False,
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
    n = len(names)
    width = 0.8 / max(n, 1)

    # (same visuals as your existing plotter)
    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, name in enumerate(names):
        vals = [histograms[name].get(k, 0.0) for k in all_k]
        bars = ax.bar([x + idx * width for x in range(len(all_k))], vals, width=width, label=name)

        # annotate values
        for rect, y in zip(bars, vals):
            label = f"{y:.2f}" if normalize else f"{int(y)}"
            ax.annotate(
                label,
                xy=(rect.get_x() + rect.get_width() / 2, y),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks([x + (n - 1) * width / 2 for x in range(len(all_k))])
    ax.set_xticklabels([str(k) for k in all_k])
    ax.set_xlabel("# flips per path (k)")
    ax.set_ylabel("Proportion" if normalize else "Count")
    ax.set_title(title or "Flips-per-path histogram")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    if save_path:
        Path(os.path.dirname(save_path) or ".").mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close(fig)


# ----------------- new helpers for decile plotting ----------------------------

def deciles_file() -> str:
    # Matches the writer name in the new decile script
    return os.path.join(
        f"data/{DATASET_NAME}/analysis/decile_distribution/by_{DISTANCE_MODE}",
        f"{DATASET_NAME}_flip_distribution_STATS_by_{DISTANCE_MODE}.json"
    )


def load_deciles(keys: List[str], field: str = "global_proportion", normalize_abs: bool = False) -> Dict[str, List[float]]:
    """
    Load decile distributions for specific keys from the single combined decile JSON.
    - field: one of the keys inside each per_index_set dict (e.g., "global_proportion", "avg_per_path", "abs_counts")
    - normalize_abs: if True and field == "abs_counts", normalize counts to proportions
    Returns: {key -> [values for deciles 0..9]}
    """
    path = deciles_file()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Combined decile file not found: {path}\n"
            "Did you run the script that writes the combined decile JSON?"
        )

    with open(path, "r") as f:
        data = json.load(f)

    per_index_set = data.get("per_index_set", {})
    out = {}
    for k in keys:
        if k not in per_index_set:
            print(f"[warn] Missing key in combined deciles: {k}")
            continue

        raw = per_index_set[k].get(field, {})
        # todo: delete this functionality. confusing
        if field == "abs_counts" and normalize_abs:
            total = sum(raw.values()) or 1
            raw = {str(dd): raw.get(str(dd), 0) / total for dd in map(str, range(10))}

        out[k] = [float(raw.get(str(d), 0.0)) for d in range(10)]

    return out


def plot_deciles_from_dict(
    deciles: Dict[str, List[float]],
    field: str = "global_proportion",
    normalize_abs: bool = False,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Plot decile distributions from dict {label -> [values for deciles 0..9]}.
    """
    if not deciles:
        raise ValueError("No decile data provided.")

    xs = list(range(10))
    names = list(deciles.keys())
    n = len(names)
    width = 0.8 / max(n, 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, name in enumerate(names):
        ys = deciles[name]
        xpos = [x + idx * width for x in xs]
        bars = ax.bar(xpos, ys, width=width, label=name)

        for rect, y in zip(bars, ys):
            if field == "abs_counts" and not normalize_abs:
                label = f"{int(round(y))}"
            else:
                label = f"{y:.2f}"
            ax.annotate(
                label,
                xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks([x + (n - 1) * width / 2 for x in xs])
    ax.set_xticklabels([f"{10*d}-{10*(d+1)}%" for d in xs])
    ax.set_xlabel("Edit-path decile (relative position)")
    ax.set_ylabel("Proportion" if field != "abs_counts" or normalize_abs else "Count")
    ax.set_title(title or f"Decile distribution ({field})")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    if save_path:
        Path(os.path.dirname(save_path) or ".").mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close(fig)


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

    # plot
    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, name in enumerate(names):
        vals = [proc[name].get(k, 0) for k in all_k]
        bars = ax.bar([x + idx * width for x in range(len(all_k))], vals, width=width, label=name)

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
    ax.set_xlabel("# flips per path (k)")
    ax.set_ylabel("Proportion" if normalize else "Count")
    ax.set_title(title or "Flips-per-path histogram")
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
    ax.set_xlabel("Edit-path decile (relative position)")
    ax.set_ylabel("Proportion" if field != "abs_counts" or normalize_abs else "Count")
    ax.set_title(title or f"Decile distribution ({field})")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    if save_path:
        Path(os.path.dirname(save_path) or ".").mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close(fig)
