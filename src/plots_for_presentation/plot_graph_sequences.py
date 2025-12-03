import os
import json
import pickle
import math
from pathlib import Path
from typing import List
import networkx as nx
import matplotlib.pyplot as plt

from config import DATASET_NAME, ROOT

# --- Config ---

SELECTED_DIR = os.path.join(ROOT, DATASET_NAME, "paths_to_plot")
PLOT_DIR = os.path.join(ROOT, DATASET_NAME, "path_plots")
LAYOUT = "kamada_kawai"  # "spring" | "kamada_kawai" | "spectral" | "circular" | "shell"
NODE_ATTR = "primary_label"  # None to use node IDs
EDGE_ATTR = "label"  # None to hide edge labels

os.makedirs(PLOT_DIR, exist_ok=True)


# --- Helpers ---

def load_sequence(path: str) -> List[nx.Graph]:
    with open(path, "rb") as f:
        seq = pickle.load(f)
    if not isinstance(seq, (list, tuple)) or not all(isinstance(g, nx.Graph) for g in seq):
        raise ValueError(f"{path} does not contain a list of networkx Graphs")
    return list(seq)


def compute_layout(G: nx.Graph, layout: str):
    if layout == "spring":
        return nx.spring_layout(G, seed=0)
    if layout == "kamada_kawai":
        return nx.kamada_kawai_layout(G)
    if layout == "spectral":
        return nx.spectral_layout(G)
    if layout == "circular":
        return nx.circular_layout(G)
    if layout == "shell":
        return nx.shell_layout(G)
    return nx.spring_layout(G, seed=0)


def plot_graph_sequence(
    seq: List[nx.Graph],
    save_path: str,
    title: str = "",
    layout: str = "spring",
    node_label_attr: str = "primary_label",
    font_size: int = 12,
    node_size: int = 800,
    ncols: int = 2,   # number of subplot columns
):
    if not seq:
        return

    # Consistent layout across sequence
    if layout == "spring":
        pos = nx.spring_layout(seq[0], seed=0)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(seq[0])
    elif layout == "spectral":
        pos = nx.spectral_layout(seq[0])
    elif layout == "circular":
        pos = nx.circular_layout(seq[0])
    else:
        pos = nx.spring_layout(seq[0], seed=0)

    n = len(seq)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(6*ncols, 6*nrows),
        dpi=120
    )
    axes = axes.flatten() if n > 1 else [axes]

    for k, (G, ax) in enumerate(zip(seq, axes)):

        # Build labels as "id:label"
        node_labels = {
            n: f"{n}:{d.get(node_label_attr, '?')}" for n, d in G.nodes(data=True)
        }

        nx.draw(
            G, pos=pos, ax=ax,
            with_labels=False, node_size=node_size, alpha=0.9
        )
        nx.draw_networkx_labels(G, pos=pos, labels=node_labels, font_size=font_size, ax=ax)

        if EDGE_ATTR is not None:
            edge_labels = nx.get_edge_attributes(G, EDGE_ATTR)
            if edge_labels:  # only draw if labels exist
                nx.draw_networkx_edge_labels(
                    G, pos=pos, edge_labels=edge_labels,
                    font_size=10, ax=ax
                )
            else:
                print("'edge_labels' is None. Printing edge labels skipped.")
        edit_step = G.graph.get("edit_step", k)
        ax.set_title(f"Edit Step {edit_step}:", loc='left', fontsize=14, fontweight="bold")
        ax.axis("off")

    # remove any axes if seq < nrows*ncols
    for ax in axes[len(seq):]:
        ax.axis("off")

    if title:
        fig.suptitle(title, y=0.98, fontsize=14)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot → {save_path}")


# ---- Run ----

if __name__ == "__main__":

    manifest_path = os.path.join(SELECTED_DIR, "selected_sequences_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    selections = manifest.get("selections", {})
    for kind in ["same", "diff"]:
        sel = selections.get(kind)
        if not sel:
            print(f"[{kind.upper()}] no selection present")
            continue

        seq_path = sel["sequence_path"]
        meta = sel["meta"]
        seq = load_sequence(seq_path)
        out_path = os.path.join(PLOT_DIR, f"{kind}_g{meta['i']}_to_g{meta['j']}_it{meta['it']}.png")
        # use classes from meta
        src_cls = meta.get("source_class", "?")
        tgt_cls = meta.get("target_class", "?")
        if kind == "same":
            title_prefix = "Zwischen Graphen gleicher Klasse"
        else:
            title_prefix = "Zwischen Graphen verschiedener Klasse"

        title = (
            f"{title_prefix} ({src_cls} → {tgt_cls})"
        )
        plot_graph_sequence(seq, save_path=out_path, title=None, layout=LAYOUT)
