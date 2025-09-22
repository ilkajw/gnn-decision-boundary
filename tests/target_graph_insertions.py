import json
import os
import math
from pathlib import Path
from config import DATASET_NAME, ROOT, MODEL
from index_sets_utils import build_index_set_cuts, cut_pairs, graphs_correctly_classified, pairs_within

if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    file_path = os.path.join(PROJECT_ROOT, ROOT, DATASET_NAME, "test", f"{DATASET_NAME}_paths_with_target_graph_inserted.json")
    with open(file_path, "r") as f:
        raw = json.load(f)
    paths = {tuple(p) for p in raw}

    corr = graphs_correctly_classified(dataset_name=DATASET_NAME, model=MODEL)
    corr_pairs = {tuple(p) for p in pairs_within(corr)}
    
    paths = cut_pairs(paths, corr_pairs)
    print(type(paths))
    idx_sets = build_index_set_cuts()

    data = {"abs": {}, "rel": {}}

    data["abs"]["total"] = len(paths)
    data["rel"]["total"] = len(paths)/math.comb(len(corr), 2)

    for key in idx_sets.keys():
        pairs = idx_sets[key]
        data["abs"][key] = len(cut_pairs(pairs, paths))
        data["rel"][key] = len(cut_pairs(pairs, paths)) / len(pairs)

    print(data)

    with open(os.path.join(ROOT, DATASET_NAME, "test",
                           f"{DATASET_NAME}_target_graph_insertions_per_idxset.json"), "w") as f:
        json.dump(data, f, indent=2)
