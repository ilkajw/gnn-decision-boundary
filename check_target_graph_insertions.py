import json

from config import DATASET_NAME
from index_sets_utils import build_index_set_cuts, cut_pairs, graphs_correctly_classified, pairs_within

if __name__ == "__main__":

    with open(f"data_control/{DATASET_NAME}/analysis/{DATASET_NAME}_paths_with_target_graph_inserted.json", "r") as f:
        raw = json.load(f)
    paths = {tuple(p) for p in raw}

    corr = graphs_correctly_classified()
    corr_pairs = {tuple(p) for p in pairs_within(corr)}
    
    paths = cut_pairs(paths, corr_pairs)
    print(type(paths))
    idx_sets = build_index_set_cuts()

    data = {"abs": {}, "rel": {}}

    data["abs"]["total"] = len(paths)
    data["rel"]["total"] = len(paths)/12720  # 160 correctly classified -> bin(160, 2) paths total

    for key in idx_sets.keys():
        pairs = idx_sets[key]
        data["abs"][key] = len(cut_pairs(pairs, paths))
        data["rel"][key] = len(cut_pairs(pairs, paths)) / len(pairs)

    print(data)

    with open(f"data_control/{DATASET_NAME}/analysis/{DATASET_NAME}_target_graph_insertions_per_idxset.json", "w") as f:
        json.dump(data, f, indent=2)
