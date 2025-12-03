# read_ops_heatmaps.py

"""
Generate operation-by-decile heatmaps from per-num-flips analysis results.

Reads the consolidated JSON at ANALYSIS_DIR/<DATASET_NAME>_<MODEL>_flip_distribution_per_num_flips_by_<DISTANCE_MODE>.json,
extracts per-index-set operation shares by decile, and writes PNG heatmaps under
ANALYSIS_DIR/plots/ops_heatmaps. Rows = operations, columns = deciles (0–10% … 90–100%).

Requires config variables: ANALYSIS_DIR, DATASET_NAME, MODEL, DISTANCE_MODE.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

from config import ANALYSIS_DIR, DATASET_NAME, MODEL, DISTANCE_MODE

# ---- Settings ----
DECILES_KEYS = [str(i) for i in range(10)]
DECILES_LABELS = [f"{i*10}-{(i+1)*10}%" for i in range(10)]
ANNOTATE_PERCENTAGES = True    # set False to hide numbers on the heatmap
CMAP = "Blues"              # choose matplotlib colormap
COLOR_RANGE = (0.0, 1.0)       # shares in [0,1] for color scale

json_path = os.path.join(
    ANALYSIS_DIR,
    f"{DATASET_NAME}_{MODEL}_flip_distribution_per_num_flips_by_{DISTANCE_MODE}.json"
)

outdir = os.path.join(ANALYSIS_DIR, "plots", "ops_heatmaps")
os.makedirs(outdir, exist_ok=True)


# ---- Helpers ----

def _to_float_safe(x):
    """Convert x to float robustly (handles '00.2539', None, etc.)."""
    if x is None:
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            try:
                return float(x.lstrip("0") or "0")
            except Exception:
                return np.nan
    return np.nan


def extract_ops_by_decile_table(d: Dict, index_set: str, flips: str) -> pd.DataFrame:
    """
    Returns a DataFrame with rows=operations, cols=deciles (0-10%,...,90-100%),
    values = normalized shares from ops_by_decile[decile]['norm'].
    """
    try:
        section = d["per_index_set"][index_set][flips]["ops_by_decile"]
    except KeyError as e:
        raise KeyError(
            f"Missing part in JSON for per_index_set[{index_set}][{flips}]['ops_by_decile']: {e}"
        )

    # collect all operations appearing in any decile
    ops: List[str] = []
    for dec in DECILES_KEYS:
        dec_obj = section.get(dec, {})
        norm = dec_obj.get("norm", {}) or {}
        for op in norm.keys():
            if op not in ops:
                ops.append(op)
    ops = sorted(ops)

    # build matrix
    data = []
    for op in ops:
        row = []
        for dec in DECILES_KEYS:
            val = np.nan
            dec_obj = section.get(dec, {})
            norm = dec_obj.get("norm", {}) or {}
            if op in norm:
                val = _to_float_safe(norm[op])
            row.append(val)
        data.append(row)

    df = pd.DataFrame(data, index=ops, columns=DECILES_LABELS)

    # ensure all expected columns present and ordered
    for col in DECILES_LABELS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[DECILES_LABELS]

    return df


def save_heatmap(df: pd.DataFrame, title: str, outpath: Path):
    """
    Plain-matplotlib heatmap with optional numeric annotations; saves to PNG.
    Each column is a decile; each row is an operation.
    """
    if df.isna().all().all():
        print(f"[WARN] All values are NaN for '{title}'. No heatmap saved.")
        return

    # Replace NaN with 0.0 just for display; keep a mask if needed later
    A = df.fillna(0.0).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=180)
    im = ax.imshow(
        A, aspect="auto", cmap=CMAP, vmin=COLOR_RANGE[0], vmax=COLOR_RANGE[1]
    )

    # ticks/labels
    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_yticks(np.arange(df.shape[0]))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticklabels(df.index)
    ax.set_xlabel("Edit-Pfad-Segment")
    ax.set_ylabel("Operation")
    # ax.set_title(title)

    # numeric annotations
    if ANNOTATE_PERCENTAGES:
        # values look like shares (0..1)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                val = A[i, j]
                txt = f"{100*val:.0f}%"
                color = "white" if val > 0.6 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Anteil")

    plt.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath)
    plt.close(fig)
    print(f"[OK] Saved heatmap: {outpath}")

# ---- Run -----
if __name__ == "__main__":
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks: List[Tuple[str, str, str]] = [
        # (index_set, flips, label_for_files)
        ("same_class_all", "2", "same_class_all__flips_2"),
        ("diff_class_all", "1", "diff_class_all__flips_1"),
        ("diff_class_all", "3", "diff_class_all__flips_3"),
    ]

    for index_set, flips, tag in tasks:
        print(f"\n=== {index_set} | flips={flips} ===")
        try:
            df = extract_ops_by_decile_table(data, index_set, flips)

            # optional console preview
            with pd.option_context("display.max_rows", None, "display.max_columns", None):
                print(df)

            # only save heatmap (no CSV)
            png_path = Path(outdir) / f"{tag}.png"
            save_heatmap(df, f"Operationsverteilung — {index_set}, Wechsel={flips}", png_path)

        except KeyError as e:
            print(f"[WARN] Could not extract section: {e}")
