import os
import json
import math
import matplotlib.pyplot as plt

from matplotlib.colors import to_rgb
from config import DATASET_NAME, DISTANCE_MODE, MODEL, ANALYSIS_DIR

# ----- Set input, ouput paths -----

INPUT_PATH = os.path.join(
    ANALYSIS_DIR,
    f"{DATASET_NAME}_{MODEL}_flip_distribution_per_num_flips_by_{DISTANCE_MODE}.json",
)
PLOT_DIR = os.path.join(
    ANALYSIS_DIR,
    "plots",
    "flip_distributions_ops"
)
os.makedirs(PLOT_DIR, exist_ok=True)


# ---- Function definitions ----

def _collect_ops(ops_by_decile: dict[str, dict]) -> list[str]:
    """Collect a stable, sorted list of operation labels present across all deciles."""
    ops = set()
    for d in range(10):
        dkey = str(d)
        if dkey not in ops_by_decile:
            continue
        abs_dict = ops_by_decile[dkey].get("abs", {})
        for op, cnt in abs_dict.items():
            if cnt and cnt > 0:
                ops.add(op)
    return sorted(ops)


def _lighten(color, factor=0.5):
    """Blend color toward white by given factor (0=original, 1=white)."""
    r, g, b = to_rgb(color)
    return (r + (1-r)*factor, g + (1-g)*factor, b + (1-b)*factor)

def _color_for_ops(ops: list[str]):

    color_map = {
        "remove_edge": plt.cm.tab10(3),    # red
        "add_edge": plt.cm.tab10(1),       # orange
        "remove_node": plt.cm.tab10(4),    # purple
        "add_node": plt.cm.tab10(2),       # green
        "relabel_node": plt.cm.tab10(0),   # blue
        "target_graph_insertion": (0.5, 0.5, 0.5),  # grey as RGB tuple
    }

    colors = {}
    for op in ops:
        if op in color_map:
            colors[op] = color_map[op]
        else:
            # fallback
            colors[op] = (0.8, 0.8, 0.8)
    return colors


def plot_ops_composition(entry: dict, k: int, save_path: str, description=None):
    """
    Stacked bar chart of OPERATION composition across deciles for given k.
    - Bar height = share of ALL flips in that decile (entry["norm"][decile])
    - Segment height = bar height * share(op within decile) (ops_by_decile[decile]["norm"][op])
    """
    if "ops_by_decile" not in entry or "norm" not in entry:
        print(f"[warn] missing 'ops_by_decile' or 'norm' in entry for k={k}")
        return

    ops_by_decile = entry["ops_by_decile"]
    decile_total_norm = entry["norm"]  # dict of decile -> fraction of all flips in that decile

    # Gather operations and colors
    ops = _collect_ops(ops_by_decile)
    if not ops:
        print(f"[WARN] No operations for k={k}")
        return
    op_colors = _color_for_ops(ops)

    xs = list(range(10))
    bottoms = [0.0] * 10
    fig, ax = plt.subplots(figsize=(10, 6))

    # For each op, stack its contribution across deciles
    for op in ops:
        ys = []
        for d in xs:
            dkey = str(d)
            decile_bar = float(decile_total_norm.get(dkey, 0.0))  # total height for this decile
            op_share = float(ops_by_decile.get(dkey, {}).get("norm", {}).get(op, 0.0))  # within-decile share
            ys.append(decile_bar * op_share)
        bars = ax.bar(xs, ys, bottom=bottoms, label=op, color=op_colors[op])
        bottoms = [b + y for b, y in zip(bottoms, ys)]

        # Optional: label segment with within-decile share if it's visually meaningful
        for rect, d in zip(bars, xs):
            dkey = str(d)
            op_share = float(ops_by_decile.get(dkey, {}).get("norm", {}).get(op, 0.0))
            if op_share >= 0.10 and rect.get_height() > 0.02:  # label only if >=10% and tall enough
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    rect.get_y() + rect.get_height() / 2,
                    f"{op_share:.2f}",
                    ha="center", va="center",
                    fontsize=8, color="black", weight="bold"
                )

    # Total labels at top of bars
    for x, total in zip(xs, bottoms):
        if total > 0:
            ax.text(x, total + 0.002, f"{total:.2f}", ha="center", va="bottom", fontsize=9, weight="bold")

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.12)

    # styling
    ax.set_xlabel("Edit-Pfad-Segment")
    ax.set_ylabel("Anteil Flips (gesamt) • Segmentanteile nach Operation")
    if description:
        ax.text(
            0.5, 0.97, description,
            ha="center", va="top", fontsize=11,
            bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round,pad=0.5"),
            transform=ax.transAxes
        )
    ax.legend(title="Operation", ncols=2, fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{i*10}-{(i+1)*10}%" for i in xs])
    ax.grid(axis="y", linestyle=":", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---- Run ----

if __name__ == "__main__":
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Missing input JSON: {INPUT_PATH}")

    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    all_sets = data.get("per_index_set", {})

    for set_name, stats in all_sets.items():

        print(f"→ Plotting ops-composition for index set: {set_name}")

        for k_str, entry in stats.items():

            if k_str == "num_pairs":   # skip metadata
                continue
            k = int(k_str)
            num_paths = entry.get("num_paths", "NA")

            # Subfolder per k
            k_dir = os.path.join(PLOT_DIR, f"k{k}")
            os.makedirs(k_dir, exist_ok=True)

            out_path = os.path.join(
                k_dir, f"{DATASET_NAME}_{MODEL}_{set_name}_ops_composition_k{k}.png"
            )
            plot_ops_composition(
                entry,
                k,
                out_path,
                description=f"{set_name} (n={num_paths})"
            )
            print(f"Saved plot → {out_path}")
