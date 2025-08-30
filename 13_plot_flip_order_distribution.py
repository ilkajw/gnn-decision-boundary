import os
import json
import matplotlib.pyplot as plt
from config import DATASET_NAME, DISTANCE_MODE, MODEL, ANALYSIS_DIR

# todo: delete calculation and plot for global as it's equal to 'diff_class_all' or 'same_class_all'


# ----- paths ----
INPUT_PATH = os.path.join(
    ANALYSIS_DIR,
    f"{DATASET_NAME}_{MODEL}_flip_distribution_per_num_flips_by_{DISTANCE_MODE}.json",
)
PLOT_DIR = os.path.join(
    ANALYSIS_DIR,
    "plots",
    "flip_distributions"
)
os.makedirs(PLOT_DIR, exist_ok=True)


# ---- plotting ----

def plot_flip_order_distribution(entry: dict, k: int, save_path: str, description=None):
    """
    Stacked bar chart of flip orders across deciles for given k.
    Labels:
      - each color segment: fraction of that flip-order's flips in the decile
      - top of bar: fraction of all flips in that decile
    """
    if "flip_order_distribution" not in entry:
        print(f"[WARN] no flip_order_distribution for k={k}")
        return

    flip_order_dist = entry["flip_order_distribution"]
    n_orders = len(flip_order_dist)
    xs = list(range(10))
    bottoms = [0.0] * 10

    # --- totals per flip order and global ---
    totals_per_order = {
        order: sum(flip_order_dist[str(order + 1)].values())
        for order in range(n_orders)
    }
    total_all = sum(totals_per_order.values())

    # color palette (blue, yellow, grey)
    colors = ['#808080', "#1f77b4", "#f2c94c"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # --- stack per flip order ---
    for order in range(n_orders):
        order_key = str(order + 1)  # flip name: 1, 2, 3, ...
        raw = [flip_order_dist[order_key][str(d)] for d in xs]
        order_total = totals_per_order[order]

        # bar heights = share of all flips
        ys = [val / total_all if total_all > 0 else 0.0 for val in raw]

        bars = ax.bar(xs, ys,
                      bottom=bottoms,
                      label=f"Flip {order + 1}",
                      color=colors[(order+1) % len(colors)]
                      )
        # only show per-flip oder distribution values if k>1. for k=1 total bar value is equal
        if k > 1:
            # segment labels = share of that flip-order’s flips
            for rect, raw_val in zip(bars, raw):
                if order_total > 0:
                    frac_order = raw_val / order_total
                else:
                    frac_order = 0.0
                if frac_order > 0.01:
                    ax.text(
                        rect.get_x() + rect.get_width() / 2,
                        rect.get_y() + rect.get_height() / 2,
                        f"{frac_order:.2f}",
                        ha="center", va="center",
                        fontsize=8, color="black", weight="bold"
                    )

        bottoms = [b + y for b, y in zip(bottoms, ys)]

    # --- total labels on top of bars ---
    for x, b in zip(xs, bottoms):
        if b > 0:
            ax.text(
                x,
                b + 0.002,
                f"{b:.2f}",
                ha="center", va="bottom", fontsize=9, color="black", weight="bold"
            )

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.1)  # add 10% space on top

    # --- styling ---
    ax.set_xlabel("Edit-Pfad-Segment")
    ax.set_ylabel("Anteil Flips")
    if description:
        ax.text(
            0.5, 0.97, description,  # x, y in axes coordinates
            ha="center", va="top",
            fontsize=11,
            bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round,pad=0.5"),
            transform=ax.transAxes  # axes coordinates (0..1)
        )
    ax.legend()
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{i * 10}-{(i + 1) * 10}%" for i in xs])  # x axis: 0-10% ... 90-100%
    ax.grid(axis="y", linestyle=":", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# ------------ run -----------------

if __name__ == "__main__":

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Missing input JSON: {INPUT_PATH}")

    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    # go through global + all index sets
    all_sets = {"global": data["global"], **data["per_index_set"]}

    for set_name, stats in all_sets.items():
        print(f"→ Plotting for index set: {set_name}")
        for k_str, entry in stats.items():
            if k_str == "num_pairs":   # skip metadata
                continue
            k = int(k_str)
            num_paths = entry.get("num_paths", "NA")

            # create subfolder for this k
            k_dir = os.path.join(PLOT_DIR, f"k{k}")
            os.makedirs(k_dir, exist_ok=True)

            out_path = os.path.join(k_dir, f"{DATASET_NAME}_{MODEL}_{set_name}_flip_order_distribution_k{k}.png")
            plot_flip_order_distribution(
                entry,
                k,
                out_path,
                description=f"{set_name} (n={num_paths})"
            )
            print(f"Saved plot → {out_path}")
