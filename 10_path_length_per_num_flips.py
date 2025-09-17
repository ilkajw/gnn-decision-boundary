import os
import json
import statistics as stats
from collections import defaultdict

from config import DATASET_NAME, ANALYSIS_DIR, DISTANCES_PATH, FLIPS_PATH

# TODO:  calculate with numpy

# ---- Function definition -----
def safe_stdev(values):
    return float(stats.stdev(values)) if len(values) > 1 else 0.0

def main():
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

    header = f"{'k':>4}  {'count':>6}  {'mean':>10}  {'median':>10}  {'stddev':>10}  {'max':>10}"
    print(header)
    print("-" * len(header))

    for k in sorted(per_k):
        vals = per_k[k]
        mean_v = float(stats.mean(vals))
        med_v = float(stats.median(vals))
        std_v = safe_stdev(vals)
        max_v = float(max(vals))
        print(f"{k: >4}  {len(vals): >6}  {mean_v: >10.4f}  {med_v: >10.4f}  {std_v: >10.4f}  {max_v: >10.4f}")

    # todo: save
# ---- run ----
if __name__ == "__main__":
    main()
