import os
import json
import statistics as stats
from collections import defaultdict

from config import DATASET_NAME, ANALYSIS_DIR, DISTANCES_PATH, FLIPS_PATH, MODEL, DISTANCE_MODE, CORRECTLY_CLASSIFIED_ONLY

# TODO:  calculate with numpy
# ---- Set output path ----

output_path = os.path.join(ANALYSIS_DIR, f"{DATASET_NAME}_{MODEL}_path_length_stats_per_num_flips_by_{DISTANCE_MODE}.json")

# ---- Function definitions -----

def safe_stdev(values):
    return float(stats.stdev(values)) if len(values) > 1 else 0.0


def calculate_stats(output_path):
    with open(FLIPS_PATH, "r") as f:
        flips = json.load(f)
    with open(DISTANCES_PATH, "r") as f:
        dists = json.load(f)

    per_k = defaultdict(list)

    for key, flip_ops in flips.items():
        k = len(flip_ops)
        if key in dists:
            length = dists[key]
            per_k[k].append(length)

    header = f"{'k': >4}  {'count': >6}  {'mean': >10}  {'median': >10}  {'stddev': >10}  {'max': >10}"
    print(header)
    print("-" * len(header))

    summary = {}

    for k in sorted(per_k):
        vals = per_k[k]
        mean_v = float(stats.mean(vals))
        med_v = float(stats.median(vals))
        std_v = safe_stdev(vals)
        max_v = float(max(vals))
        summary[f"{k}"] = {"mean": mean_v,
                           "std": std_v,
                           "median": med_v,
                           "max": max_v,
                           "n": len(vals),
                           "vals": vals

                           }

        print(f"{k: >4}  {len(vals): >6}  {mean_v: >10.4f}  {med_v: >10.4f}  {std_v: >10.4f}  {max_v: >10.4f}")

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)


# ---- Run ----

if __name__ == "__main__":

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    calculate_stats(output_path=output_path)
