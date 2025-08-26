import json
import os
from pathlib import Path
from typing import List, Dict, Optional

from matplotlib import pyplot as plt

from config import DATASET_NAME, DISTANCE_MODE

# define input, output paths
ANALYSIS_DIR = f"data_control/{DATASET_NAME}/analysis"
PLOT_DIR = os.path.join(ANALYSIS_DIR, "plots", "decile_distribution")
os.makedirs(PLOT_DIR, exist_ok=True)

# ------------ helpers --------------

def deciles_file() -> str:
    return os.path.join(
        f"data_control/{DATASET_NAME}/analysis/decile_distribution/by_{DISTANCE_MODE}",
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
    ax.set_xlabel("Dezil Edit-Pfad")
    ax.set_ylabel("Anteil Pfade" if field != "abs_counts" or normalize_abs else "Anzahl Pfade")
    # ax.set_title(title or f"Flip-Verteilung entlang Edit-Pfad-Dezilen")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    if save_path:
        Path(os.path.dirname(save_path) or ".").mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":

    # -------------- same vs. diff --------

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
