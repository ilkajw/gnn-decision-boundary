import os
import json
import numpy as np
from scipy import stats
from pathlib import Path

from config import ANALYSIS_DIR, DATASET_NAME, MODEL, DISTANCE_MODE

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Load JSON ---
with open(os.path.join(PROJECT_ROOT, ANALYSIS_DIR, f"{DATASET_NAME}_{MODEL}_path_length_stats_per_num_flips_by_{DISTANCE_MODE}.json"), "r") as f:
    data = json.load(f)

# --- Build full arrays of (k, length) ---
ks = []
lengths = []

for k_str, group in data.items():
    k_val = int(k_str)
    vals = group["vals"]
    ks.extend([k_val] * len(vals))
    lengths.extend(vals)

ks = np.array(ks)
lengths = np.array(lengths)

print(f"Total samples: {len(lengths)}")

# --- Pearson correlation test ---
r, p = stats.pearsonr(ks, lengths)
print("Pearson correlation:")
print(f"  r = {r:.4f}, p = {p:.4e}")

# --- Spearman correlation test ---
rho, p_s = stats.spearmanr(ks, lengths)
print("\nSpearman correlation:")
print(f"  rho = {rho:.4f}, p = {p_s:.4e}")
