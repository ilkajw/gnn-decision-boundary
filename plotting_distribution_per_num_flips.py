# plot_num_flips_deciles.py
import os
import json
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt

from config import DATASET_NAME, DISTANCE_MODE

# ------------------------ paths -----------------------------

ANALYSIS_DIR = f"data/{DATASET_NAME}/analysis"
IN_PATH = os.path.join(
    ANALYSIS_DIR,
    f"paths_per_num_flips/{DISTANCE_MODE}",
    f"{DATASET_NAME}_flip_distribution_per_num_flips_by_{DISTANCE_MODE}.json",
)
PLOT_DIR = os.path.join(ANALYSIS_DIR, "plots", "num_flips_deciles", DISTANCE_MODE)
os.makedirs(PLOT_DIR, exist_ok=True)


# ---------------- loaders for your JSON shape ----------------

def _k_entry(rec: dict, k: int) -> Optional[dict]:
    """Return the entry for k (keys are strings '1', '2', ...)."""
    return rec.get(str(k))


def _deciles_from_entry(entry: dict, field: str) -> List[float]:
    """Extract deciles 0..9 for a given field ('avg_proportion' or 'abs_counts')."""
    if entry is None:
        return [0.0] * 10
    block = entry.get(field, {})
    # json keys are "0"..."9"
    vals = [block.get(str(d), 0) for d in range(10)]
    # cast to float for plotting
    return [float(v) for v in vals]


def load_deciles_for_keys_at_k(
    json_path: str,
    keys: List[str],
    k: int,
    field: str = "avg_proportion",   # or "abs_counts"
    from_global: bool = False,
) -> Dict[str, List[float]]:
    """
    Load deciles for a fixed k from the combined JSON.
    - If from_global=True, read from data['global'].
    - Else, read from data['per_index_set'][key].
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Combined per-num-flips file not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    out: Dict[str, List[float]] = {}
    if from_global:
        root = data.get("global", {})
        entry = _k_entry(root, k)
        out["global"] = _deciles_from_entry(entry, field)
        return out

    per_idx = data.get("per_index_set", {})
    for name in keys:
        rec = per_idx.get(name)
        if rec is None:
            print(f"[warn] Missing index set: {name}")
            continue
        entry = _k_entry(rec, k)
        if entry is None:
            print(f"[warn] No entry for k={k} in {name}")
            continue
        out[name] = _deciles_from_entry(entry, field)
    return out


# ------------------------- plotting --------------------------

def plot_deciles_from_dict(
    deciles: Dict[str, List[float]],
    field: str,
    title: str,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Plot grouped bars for deciles 0..9 for each series in `deciles`.
    `field` controls labeling (avg_proportion => proportions, abs_counts => counts).
    Uses fixed colors (blue, orange, yellow).
    """
    if not deciles:
        raise ValueError("No decile data to plot.")

    xs = list(range(10))
    names = list(deciles.keys())
    n = len(names)
    width = 0.8 / max(n, 1)

    # color palette (blue, grey, yellow)
    colors = ["#1f77b4", '#808080', "#f2c94c"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, name in enumerate(names):
        ys = deciles[name]
        xpos = [x + idx * width for x in xs]

        bars = ax.bar(
            xpos,
            ys,
            width=width,
            label=name,
            color=colors[idx % len(colors)],  # assign series color
            linewidth=0.5,
        )

        for rect, y in zip(bars, ys):
            label = (
                f"{y:.2f}" if field == "avg_proportion"
                else (f"{int(round(y))}" if y >= 1 else f"{y:.2f}")
            )
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
    ax.set_xlabel("Edit-Pfad-Segment")
    ax.set_ylabel("Anteil Pfade" if field == "avg_proportion" else "Anzahl Pfade")
    #if title:
    #    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    if save_path:
        Path(os.path.dirname(save_path) or ".").mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close(fig)


# --------------------------------------- run plots -------------------------------------------

if __name__ == "__main__":

    FIELD = "avg_proportion"  # or "abs_counts"

    # todo: the next section is not needed anymore as we got an extra function for this
    # ----------------------------- same_class_all for k=2 -------------------------------------
    k = 2
    same_all_k2 = load_deciles_for_keys_at_k(IN_PATH, ["same_class_all"], k=k, field=FIELD)
    plot_deciles_from_dict(
        deciles=same_all_k2,
        field=FIELD,
        title=f"{DATASET_NAME}: deciles for k={k} (same_class_all, {FIELD})",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_k{k}_{FIELD}_same_all.png"),
    )

    # --------------------same: train–train vs test–test vs train–test for k=2 -----------------
    same_keys = ["same_train_train", "same_test_test", "same_train_test"]
    same_k2 = load_deciles_for_keys_at_k(IN_PATH, same_keys, k=k, field=FIELD)
    plot_deciles_from_dict(
        deciles=same_k2,
        field=FIELD,
        title=f"{DATASET_NAME}: deciles for k={k} (same: train/test/train–test, {FIELD})",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_k{k}_{FIELD}_same_train_vs_test.png"),
    )

    # ------------------- diff: train–train vs test–test vs train–test for k=1 ------------------
    k = 1
    diff_keys = ["diff_train_train", "diff_test_test", "diff_train_test"]
    diff_k2 = load_deciles_for_keys_at_k(IN_PATH, diff_keys, k=k, field=FIELD)
    plot_deciles_from_dict(
        deciles=diff_k2,
        field=FIELD,
        title=f"Distribution over path deciles for k={k} flip",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_k{k}_{FIELD}_diff_train_vs_test.png"),
    )

    # -------------------------------- diff for k=1 ---------------------------------

    diff_k1 = load_deciles_for_keys_at_k(IN_PATH, ["diff_class_all"], k=k, field=FIELD)
    plot_deciles_from_dict(
        deciles=diff_k1,
        field=FIELD,
        title=f"Distribution over path deciles for k={k} flip",
        save_path=os.path.join(PLOT_DIR, f"{DATASET_NAME}_k{k}_{FIELD}_diff_all.png"),
    )
