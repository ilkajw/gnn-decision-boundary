# plotting_two_flip_distribution.py
import json
import difflib
from pathlib import Path
import matplotlib.pyplot as plt

from config import DATASET_NAME

# ------------------------------------------------------------
# CONFIG
JSON_PATH = f"data/{DATASET_NAME}/analysis/flip_distributions/{DATASET_NAME}_first_second_flips_by_cost.json"
SAVE_PATH = f"data/{DATASET_NAME}/analysis/plots/flip_distributions"
SHOW_PLOTS = False

GROUPS_TO_PLOT = [
    "same_class_all",
    "same_class_0_all",
    "same_class_1_all",
    "same_train_train",
    "same_test_test",
    "same_train_test",
]
# ------------------------------------------------------------


if not SHOW_PLOTS:
    import matplotlib
    matplotlib.use("Agg")


def load_results(json_path: str | Path):
    with open(json_path, "r") as f:
        return json.load(f)


def extract_props(group_data, which="first"):
    props_map = group_data[which]["proportion"]
    deciles = list(range(10))
    props = [float(props_map.get(str(d), 0.0)) for d in deciles]
    return deciles, props


def plot_group(group_name, group_data, save_dir: Path | None):
    """Two bar charts (first + second flip) side by side, with value labels."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for ax, (i, which) in zip(axes, enumerate(["first", "second"], start=1)):
        dec, props = extract_props(group_data, which)
        bars = ax.bar(dec, props)
        labels = [f"{d * 10}-{(d + 1) * 10}%" for d in dec]
        ax.set_xticks(dec)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Pfad-Segment nach Kosten")
        ax.set_ylabel("Anteil Pfade")
        ax.set_title(f"Flip {i}")  # -> Flip 1 / Flip 2
        ax.grid(True, axis="y", linestyle=":", linewidth=0.5)

        # annotate each bar
        for rect, val in zip(bars, props):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height(),
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    fig.suptitle(f"Flip-Verteilung '{group_name}' (n={group_data['num_paths']})", fontsize=14)
    fig.tight_layout()

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / f"{group_name}_bars.png", dpi=150, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_overlay(group_name, group_data, save_dir: Path | None):
    """Overlay first vs second proportions, with value labels."""
    dec, p_first = extract_props(group_data, "first")
    _, p_second = extract_props(group_data, "second")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dec, p_first, marker="o", label="Flip 1")
    ax.plot(dec, p_second, marker="o", label="Flip 2")

    # add text labels for each point
    for x, y in zip(dec, p_first):
        ax.text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=8)
    for x, y in zip(dec, p_second):
        ax.text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(dec)
    ax.set_xlabel("Dezil")
    ax.set_ylabel("Anteil Pfade")
    ax.set_title(f"{group_name} (n={group_data['num_paths']})")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5)
    ax.legend()
    fig.tight_layout()

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / f"{group_name}_overlay.png", dpi=150, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    results = load_results(JSON_PATH)
    all_groups = list(results.keys())
    save_dir = Path(SAVE_PATH) if SAVE_PATH else None

    print("Available groups in JSON:", all_groups)
    print("Requested groups:", GROUPS_TO_PLOT)


    valid_groups = []
    for g in GROUPS_TO_PLOT:
        if g in results:
            valid_groups.append(g)
        else:
            suggestion = difflib.get_close_matches(g, all_groups, n=1)
            if suggestion:
                print(f"⚠️  Group '{g}' not found. Did you mean '{suggestion[0]}'?")
            else:
                print(f"⚠️  Group '{g}' not found. Skipping.")

    if not valid_groups:
        raise SystemExit("No valid groups to plot. Please fix GROUPS_TO_PLOT.")

    for g in valid_groups:
        data = results[g]
        plot_group(g, data, save_dir=save_dir)
        plot_overlay(g, data, save_dir=save_dir)

    print(f"Done. Figures saved to: {save_dir.resolve() if save_dir else '(not saved)'}")
