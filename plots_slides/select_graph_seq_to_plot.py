import os
import re
import glob
import json
import pickle
import networkx as nx
from typing import Dict, List, Tuple, Optional
from torch_geometric.datasets import TUDataset

from config import DATASET_NAME, ROOT

# --- config ---

NX_INPUT_DIR = f"{ROOT}/{DATASET_NAME}/nx_edit_path_graphs"
SELECTED_DIR = f"{ROOT}/{DATASET_NAME}/selected_sequences"  # to save picked sequences to
MAX_LEN = 4
PYG_ROOT = "data/pyg"  # for TUDataset to read labels

os.makedirs(SELECTED_DIR, exist_ok=True)

# --- helpers ---

_fname_re = re.compile(r"g(\d+)_to_g(\d+)_it(\d+)_graph_sequence\.pkl$")


def load_true_labels_pyg(name: str = "MUTAG", root: str = PYG_ROOT) -> Dict[int, int]:
    ds = TUDataset(root=root, name=name)
    return {i: int(ds[i].y.item()) for i in range(len(ds))}


def parse_ids(path: str) -> Optional[Tuple[int,int,int]]:
    m = _fname_re.search(os.path.basename(path))
    if not m:
        return None
    return tuple(map(int, m.groups()))


def load_sequence(path: str) -> List[nx.Graph]:
    with open(path, "rb") as f:
        seq = pickle.load(f)
    if not isinstance(seq, (list, tuple)) or not all(isinstance(g, nx.Graph) for g in seq):
        raise ValueError(f"{path} does not contain a list of networkx Graphs")
    return list(seq)


def collect_index(nx_dir: str):
    index = []
    for p in glob.glob(os.path.join(nx_dir, "*_graph_sequence.pkl")):
        ids = parse_ids(p)
        if not ids:
            continue
        i, j, it = ids
        try:
            seq_len = len(load_sequence(p))
        except Exception:
            print(f"[warn] sequence could not be loaded for {i}, {j}")
            continue
        index.append({"path": p, "i": i, "j": j, "it": it, "len": seq_len})
    return index


def choose_sequence(index, label_map: Dict[int, int], max_len: int, same: bool):
    def is_same(i, j): return label_map.get(i) == label_map.get(j)
    candidates = [
        row for row in index
        if row["len"] <= max_len and ((is_same(row["i"], row["j"]) and same) or (not is_same(row["i"], row["j"]) and not same))
    ]
    if not candidates:
        return None
    best = max(candidates, key=lambda r: (r["len"], r["it"], r["path"]))
    best_seq = load_sequence(best["path"])
    # add class info
    i, j = best["i"], best["j"]
    best["source_class"] = label_map[i]
    best["target_class"] = label_map[j]
    return best, best_seq


def save_sequence(seq: List[nx.Graph], out_path: str):
    with open(out_path, "wb") as f:
        pickle.dump(seq, f, protocol=pickle.HIGHEST_PROTOCOL)


# --- run ---

if __name__ == "__main__":
    label_map = load_true_labels_pyg(name=DATASET_NAME)

    index = collect_index(NX_INPUT_DIR)

    same_pick = choose_sequence(index, label_map, max_len=MAX_LEN, same=True)
    diff_pick = choose_sequence(index, label_map, max_len=MAX_LEN, same=False)

    manifest = {"dataset": DATASET_NAME, "max_len": MAX_LEN, "selections": {}}

    if same_pick:
        meta, seq = same_pick
        same_out = os.path.join(SELECTED_DIR, f"same_g{meta['i']}_to_g{meta['j']}_it{meta['it']}.pkl")
        save_sequence(seq, same_out)
        manifest["selections"]["same"] = {
            "meta": meta,
            "sequence_path": os.path.abspath(same_out)
        }
        print(f"[SAME] saved → {same_out}")
    else:
        print("[SAME] none found")
        manifest["selections"]["same"] = None

    if diff_pick:
        meta, seq = diff_pick
        diff_out = os.path.join(SELECTED_DIR, f"diff_g{meta['i']}_to_g{meta['j']}_it{meta['it']}.pkl")
        save_sequence(seq, diff_out)
        manifest["selections"]["diff"] = {
            "meta": meta,
            "sequence_path": os.path.abspath(diff_out)
        }
        print(f"[DIFF] saved → {diff_out}")
    else:
        print("[DIFF] none found")
        manifest["selections"]["diff"] = None

    manifest_path = os.path.join(SELECTED_DIR, "selected_sequences_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[MANIFEST] → {manifest_path}")
