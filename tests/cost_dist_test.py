import os
import re
import json
import torch
from torch.serialization import add_safe_globals
from torch_geometric.data import Data

from config import ANALYSIS_DIR, DATASET_NAME, MODEL, MODEL_INDEPENDENT_PRECALCULATIONS_DIR

# ---------- CONFIG (edit these) ----------
SEQUENCES_DIR = r"../data_actual_best/MUTAG/GAT/predictions/edit_path_graphs_with_predictions_CUMULATIVE_COST"
DISTANCES_JSON = fr"{MODEL_INDEPENDENT_PRECALCULATIONS_DIR}\{DATASET_NAME}_dist_per_path.json"  # {"i,j": distance}
OUTPUT_JSON = fr"{ANALYSIS_DIR}/test/cost_vs_distance_mismatches.json"


# -----------------------------------------

def load_distances(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing distances JSON: {path}")
    with open(path, "r") as f:
        distances = json.load(f)
    if not isinstance(distances, dict):
        raise TypeError("Distances JSON must be an object mapping 'i,j' -> number.")
    for k, v in distances.items():
        if not isinstance(k, str):
            raise TypeError(f"Distances key must be str 'i,j', got: {type(k)}")
        if not isinstance(v, (int, float)):
            raise TypeError(f"Distances value must be number for key {k}, got: {type(v)}")
        if v <= 0:
            raise ValueError(f"Non-positive distance for {k}: {v}")
    return distances

def parse_filename(fname: str):
    m = re.fullmatch(r"g(\d+)_to_g(\d+)_it(\d+)_graph_sequence\.pt", fname)
    if not m:
        raise RuntimeError(f"Unexpected filename format: {fname}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))

def check_sequences_against_distances(sequences_dir: str, distances_map: dict, out_json: str):
    if not os.path.exists(sequences_dir):
        raise FileNotFoundError(f"Missing sequences dir: {sequences_dir}")

    add_safe_globals([Data])

    mismatches = []
    checked = 0

    for fname in sorted(os.listdir(sequences_dir)):
        if not fname.endswith(".pt"):
            continue

        i, j, k = parse_filename(fname)
        key = f"{i},{j}"
        if key not in distances_map:
            raise KeyError(f"Distance missing for key {key} (no direction swapping). File: {fname}")
        distance = distances_map[key]
        if not isinstance(distance, (int, float)) or distance <= 0:
            raise ValueError(f"Invalid distance for {key}: {distance}")

        path = os.path.join(sequences_dir, fname)
        seq = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(seq, (list, tuple)) or len(seq) == 0:
            raise RuntimeError(f"Loaded sequence is empty or not list/tuple: {fname}")
        if not all(isinstance(g, Data) for g in seq):
            raise TypeError(f"Sequence contains non-Data entries: {fname}")

        last = seq[-1]
        if not hasattr(last, "cumulative_cost"):
            raise AttributeError(f"'cumulative_cost' missing on last graph for file {fname}")
        last_cost = float(last.cumulative_cost)

        diff = last_cost - distance
        if abs(diff) > 1e-8:  # tolerate tiny float noise
            diff_share = diff / distance
            mismatches.append({
                "source_idx": i,
                "target_idx": j,
                "iteration": k,
                "last_cumulative_cost": last_cost,
                "distance": float(distance),
                "difference": diff,
                "difference_share_of_distance": diff_share,
                "file": fname,
            })
        print(f"{i}, {j} done")
        checked += 1

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({
            "checked": checked,
            "num_mismatches": len(mismatches),
            "mismatches": mismatches
        }, f, indent=2)

    print(f"Checked sequences: {checked}")
    print(f"Mismatches: {len(mismatches)}")
    print(f"Saved JSON -> {out_json}")

if __name__ == "__main__":
    distances = load_distances(DISTANCES_JSON)
    check_sequences_against_distances(
        sequences_dir=SEQUENCES_DIR,
        distances_map=distances,
        out_json=OUTPUT_JSON,
    )
