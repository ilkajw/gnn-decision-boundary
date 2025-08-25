import os, json
from config import DATASET_NAME, DISTANCE_MODE
from index_sets_utils import build_index_set_cuts

K_ALLOWED = {1, 2}
INSERTED_FILE = f"data_control/{DATASET_NAME}/analysis/{DATASET_NAME}_paths_with_target_graph_inserted.json"
OUT_PATH = f"data_control/{DATASET_NAME}/analysis/{DATASET_NAME}_target_inserts_among_last_decile_flip_k12.json"

def norm_pair(i, j):
    i, j = int(i), int(j)
    return (i, j) if i <= j else (j, i)

def load_pair_set(obj):
    """Accepts list of [i,j] / 'i,j' / dict keys / {'pairs':[...]} → set of (i,j)."""
    S = set()
    items = obj.get("pairs", list(obj.keys())) if isinstance(obj, dict) else obj
    for it in items:
        if isinstance(it, (list, tuple)) and len(it) == 2:
            S.add(norm_pair(it[0], it[1]))
        elif isinstance(it, str) and "," in it:
            a, b = it.split(",", 1); S.add(norm_pair(a, b))
        elif isinstance(it, dict) and {"i", "j"} <= set(it):
            S.add(norm_pair(it["i"], it["j"]))
    return S

def decile_of(step, dist):
    if not dist or dist <= 0: return None
    return max(0, min(int(min((float(step)/float(dist))*10.0, 9)), 9))

def has_last_decile(flips, dist):
    """True iff ANY flip lands in decile 9."""
    steps = []
    for f in flips:
        if isinstance(f, (list, tuple)) and f:
            steps.append(float(f[0]))
        elif isinstance(f, dict):
            for k in ("step", "cumulative_cost", "edit_step"):
                if k in f: steps.append(float(f[k])); break
    return any(decile_of(s, dist) == 9 for s in steps)

if __name__ == "__main__":
    # pick inputs based on distance mode
    if DISTANCE_MODE == "cost":
        dist_path  = f"data_control/{DATASET_NAME}/analysis/{DATASET_NAME}_dist_per_path.json"
        flips_path = f"data_control/{DATASET_NAME}/analysis/{DATASET_NAME}_flip_occurrences_per_path_by_cost.json"
    elif DISTANCE_MODE == "num_ops":
        dist_path  = f"data_control/{DATASET_NAME}/analysis/{DATASET_NAME}_num_ops_per_path.json"
        flips_path = f"data_control/{DATASET_NAME}/analysis/{DATASET_NAME}_flip_occurrences_per_path_by_edit_step.json"
    else:
        print(f"[WARN] DISTANCE_MODE={DISTANCE_MODE!r}; assuming 'cost'.")
        dist_path  = f"data_control/{DATASET_NAME}/analysis/{DATASET_NAME}_dist_per_path.json"
        flips_path = f"data_control/{DATASET_NAME}/analysis/{DATASET_NAME}_flip_occurrences_per_path_by_cost.json"

    with open(dist_path, "r") as f:        distances = json.load(f)
    with open(flips_path, "r") as f:       flips_per_path = json.load(f)
    with open(INSERTED_FILE, "r") as f:    inserted_pairs = load_pair_set(json.load(f))

    def get_dist(i, j):
        s1, s2 = f"{i},{j}", f"{j},{i}"
        return distances.get(s1, distances.get(s2))

    # build index sets
    idx_sets = build_index_set_cuts()
    per_idx_pairs = {key: {norm_pair(i, j) for (i, j) in pairs} for key, pairs in idx_sets.items()}

    # all paths with k∈{1,2} AND ≥1 flip in last decile
    last_decile_pairs = set()      # all qualifying pairs
    inserted_and_last = set()      # subset that are inserted

    for pair_str, flips in flips_per_path.items():
        if not flips: continue
        i, j = map(int, pair_str.split(","))
        p = norm_pair(i, j)

        k = len(flips)
        if k not in K_ALLOWED: continue

        dist = get_dist(i, j)
        if not dist: continue

        if has_last_decile(flips, dist):
            last_decile_pairs.add(p)
            if p in inserted_pairs:
                inserted_and_last.add(p)

    total_last = len(last_decile_pairs)
    hit_inserted = len(inserted_and_last)
    global_share = (hit_inserted / total_last) if total_last else 0.0

    per_index_set = {}
    for key, cut_pairs in per_idx_pairs.items():
        universe = last_decile_pairs & cut_pairs
        hits = inserted_and_last & cut_pairs
        t = len(universe)
        h = len(hits)
        per_index_set[key] = {
            "total_last_decile_pairs": t,
            "inserted_among_last_decile": h,
            "share": (h / t) if t else 0.0,
            "share_pct": (100.0 * h / t) if t else 0.0,
        }

    result = {
        "meta": {
            "dataset": DATASET_NAME,
            "distance_mode": DISTANCE_MODE,
            "k_allowed": sorted(K_ALLOWED),
            "inserted_source": INSERTED_FILE,
            "dist_path": dist_path,
            "flips_path": flips_path,
            "definition": "Among paths with 1 or 2 flips and ≥1 flip in decile 9, fraction that had target graph inserted.",
        },
        "global": {
            "total_last_decile_pairs": total_last,
            "inserted_among_last_decile": hit_inserted,
            "share": global_share,
            "share_pct": 100.0 * global_share if total_last else 0.0,
        },
        "per_index_set": per_index_set,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Global: {hit_inserted}/{total_last} ({result['global']['share_pct']:.2f}%). "
          f"Saved → {OUT_PATH}")
