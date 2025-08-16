import math
import os
import json
import torch

from pathlib import Path
from torch_geometric.data import InMemoryDataset, Data
from torch.serialization import add_safe_globals
from torch_geometric.data.data import DataEdgeAttr

from config import DATASET_NAME, DISTANCE_MODE


def _ensure_dir(path):
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)


class EditPathGraphsDataset(InMemoryDataset):
    """
    Builds an in-memory dataset from per-path .pt sequences (lists of PyG Data),
    assigning a label per graph according to label_mode:

      - "same_only": keep only same-class paths; y = common class (0/1)
      - "source":    y = source true label
      - "target":    y = target true label
      - "pseudo":    y = model prediction on the graph (optionally min_prob filter)
      - "interpolate": soft labels based on path progress t in [0,1]
           * interpolation="linear"  -> y_soft = (1-t)*y_src + t*y_tgt
           * interpolation="sigmoid" -> y_soft = σ(k*(t-0.5)) for 0→1, 1-σ(k*(t-0.5)) for 1→0
    """

    def __init__(
            self,
            seq_dir,
            base_pred_path,
            label_mode="same_only",
            interpolation="linear",  # used when label_mode == "interpolate"
            k_sigmoid=12.0,          # slope for sigmoid interpolation
            min_prob=None,           # used when label_mode == "pseudo"
            transform=None,
            pre_transform=None,
            drop_endpoints=True,
            verbose=True,
            small=False  # keep only paths with 0 or 1 flips
    ):
        self.seq_dir = seq_dir
        self.base_pred_path = base_pred_path
        self.label_mode = label_mode
        self.interpolation = interpolation
        self.k_sigmoid = float(k_sigmoid)
        self.min_prob = min_prob
        self.drop_endpoints = drop_endpoints
        self.verbose = verbose
        self.enforce_flip_pattern = bool(small)  # NEW

        root_dir = os.path.abspath(f"data/{DATASET_NAME}/processed/_editpath_inmem_root")
        super().__init__(root=root_dir, transform=transform, pre_transform=pre_transform)

        data_list = self._build_list()
        self.data, self.slices = self.collate(data_list)
        if self.verbose:
            print(f"[EditPathGraphsDataset] {len(data_list)} graphs from {self.seq_dir} | "
                  f"mode={self.label_mode} | flip_filter={self.enforce_flip_pattern}")

    # InMemoryDataset expects these
    @property
    def raw_file_names(self): return []

    @property
    def processed_file_names(self): return []

    def download(self): pass

    def process(self): pass

    # -------------- internal helpers -------------------------

    def _load_base_labels(self):
        with open(self.base_pred_path, "r") as f:
            base = json.load(f)
        # ensure int keys
        return {int(k): int(v["true_label"]) for k, v in base.items()}

    def _t_from_graph(self, g):
        # todo: works?
        if DISTANCE_MODE == "cost":
            dist = float(getattr(g, "distance", 0.0))
            step = float(getattr(g, "cumulative_cost", 0.0))
        else:
            dist = float(getattr(g, "num_all_ops", 0.0))
            step = float(getattr(g, "edit_step", 0.0))
        return max(0.0, min(step / dist, 1.0)) if dist > 0 else 0.0

    def _interp_soft_label(self, y_src, y_tgt, t):
        if self.interpolation == "linear":
            return (1 - t) * y_src + t * y_tgt
        elif self.interpolation == "sigmoid":
            # 0->1 uses σ(k*(t-0.5)); 1->0 mirrors it
            if y_src == y_tgt:
                return float(y_src)
            if y_src == 0 and y_tgt == 1:
                return 1.0 / (1.0 + math.exp(-self.k_sigmoid * (t - 0.5)))
            if y_src == 1 and y_tgt == 0:
                return 1.0 - 1.0 / (1.0 + math.exp(-self.k_sigmoid * (t - 0.5)))
            print(f"[WARN] Source label {y_src}, target label {y_tgt}: one is not binary.")
            return (1 - t) * y_src + t * y_tgt
        else:
            raise ValueError(f"Unknown interpolation: {self.interpolation}")

    def _label_for_graph(self, g, y_src, y_tgt):
        # optionally only use paths between same labels
        if self.label_mode == "same_only":
            assert y_src == y_tgt
            return float(y_src)
        elif self.label_mode == "source":
            return float(y_src)
        elif self.label_mode == "target":
            return float(y_tgt)
        elif self.label_mode == "pseudo":
            # use `prediction` on g
            pred = getattr(g, "prediction", None)
            if pred is None:
                return None
            if self.min_prob is not None:
                prob = float(getattr(g, "probability", 0.0))
                if prob < self.min_prob:
                    return None  # skip low-confidence samples
            return float(int(pred))
        elif self.label_mode == "interpolate":
            t = self._t_from_graph(g)
            return float(self._interp_soft_label(y_src, y_tgt, t))
        else:
            raise ValueError(f"Unknown label_mode: {self.label_mode}")

    def _count_prediction_flips(self, seq):
        """
        Count how many times the binary 'prediction' changes along the sequence.
        Uses only elements that have a 'prediction' attribute; requires at least 2.
        Returns an integer flip count, or None if insufficient data.
        """
        preds = []
        for g in seq:
            if hasattr(g, "prediction") and getattr(g, "prediction") is not None:
                try:
                    preds.append(int(getattr(g, "prediction")))
                except Exception:
                    pass
        if len(preds) < 2:
            return None
        flips = 0
        prev = preds[0]
        for p in preds[1:]:
            if p != prev:
                flips += 1
            prev = p
        return flips

    def _build_list(self):
        from torch_geometric.data import Data as PYGData
        add_safe_globals([PYGData])

        # load predictions of original dataset graphs
        base_labels = self._load_base_labels()

        # load list of graph sequence filenames
        files = sorted([f for f in os.listdir(self.seq_dir) if f.endswith(".pt")])

        # optionally keep only graphs between same classes
        keep_same_only = (self.label_mode == "same_only")

        data_list = []
        no_intermediates = []

        for fname in files:
            seq = torch.load(os.path.join(self.seq_dir, fname), weights_only=False)
            if not seq:
                print(f"[WARN] missing graph sequence expected in {fname}")
                continue

            # extract source and target label to infer labels
            g0 = seq[0]
            i = int(getattr(g0, "source_idx", -1))
            j = int(getattr(g0, "target_idx", -1))
            if i not in base_labels or j not in base_labels:
                if self.verbose: print(f"[WARN] missing base label for {i} or {j} in {fname}")
                continue
            y_src = base_labels[i]
            y_tgt = base_labels[j]

            # optional: keep only same-class sequences
            if keep_same_only and y_src != y_tgt:
                continue

            # filter for paths with 0 or 1 flips only
            if self.enforce_flip_pattern:
                flips = self._count_prediction_flips(seq)
                if flips is None:
                    if self.verbose:
                        print(f"[INFO] {fname}: cannot determine flips (need >=2 predictions). skipping path.")
                    continue
                want_flips = 0 if (y_src == y_tgt) else 1
                if flips != want_flips:
                    if self.verbose:
                        print(f"[INFO] {fname}: flips={flips}, condition={want_flips}. path filtered out.")
                    continue

            # drop source and target graph from sequence (already in original dataset)
            if self.drop_endpoints:
                if len(seq) <= 2:
                    no_intermediates.append((i, j))
                    # no edit path graphs remain -> skip this sequence
                    if self.verbose:
                        print(f"[INFO] {fname}: dropped endpoints and found no intermediate graphs. skipping path.")
                    continue
                else:
                    seq_iter = seq[1:-1]
            else:
                seq_iter = seq

            # add each path graph with its inferred label
            for g in seq_iter:
                y_soft = self._label_for_graph(g, y_src, y_tgt)
                if y_soft is None:
                    continue  # pseudo with low prob left out

                out = Data(
                    x=g.x,
                    edge_index=g.edge_index,
                    y=torch.tensor([y_soft], dtype=torch.float),
                )

                # keep metadata (copy from g if present)
                for key in (
                    "edit_step", "cumulative_cost", "source_idx", "target_idx", "iteration",
                    "distance", "prediction", "probability", "num_all_ops"
                ):
                    if hasattr(g, key):
                        setattr(out, key, getattr(g, key))
                    else:
                        print(f"[WARN] missing attribute {key} at edit step "
                              f"{getattr(g, 'edit_step', '?')} between {i}, {j}.")

                # todo: delete?
                # if some metadata only exists on g0/g_last, optional backfill:
                if not hasattr(out, "distance") and hasattr(g0, "distance"):
                    out.distance = getattr(g0, "distance")
                if not hasattr(out, "num_all_ops") and hasattr(g0, "num_all_ops"):
                     out.num_all_ops = getattr(g0, "num_all_ops")

                data_list.append(out)

        os.makedirs(f"data/{DATASET_NAME}/analysis/no_intermediates/", exist_ok=True)
        with open(f"data/{DATASET_NAME}/analysis/no_intermediates/"
                  f"{DATASET_NAME}_no_intermediate_graphs_at_dataset_build.json", "w") as f:
            json.dump(no_intermediates, f, indent=2)

        return data_list

    def save(self, output_path, meta_path=None):
        """
        Save the collated PyG dataset (data + slices) to a single .pt file.
        Optionally save JSON meta for reproducibility.
        """
        _ensure_dir(output_path)
        torch.save((self.data, self.slices), output_path)

        if meta_path:
            _ensure_dir(meta_path)
            meta = {
                "seq_dir": self.seq_dir,
                "base_pred_path": self.base_pred_path,
                "label_mode": self.label_mode,
                "interpolation": getattr(self, "interpolation", None),
                "k_sigmoid": getattr(self, "k_sigmoid", None),
                "min_prob": self.min_prob,
                "num_graphs": self.len(),
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
        if self.verbose:
            print(f"[EditPathGraphsDataset] Saved dataset to {output_path}"
                  f"{' and meta to ' + meta_path if meta_path else ''}")


class FlatGraphDataset(InMemoryDataset):
    """
    Lightweight dataset to load a previously saved (data, slices) .pt file
    to avoid rebuilding from sequences for training/evaluation.
    """

    def __init__(self, saved_path, transform=None, pre_transform=None, verbose=True):
        self.saved_path = saved_path
        self.verbose = verbose
        root_dir = os.path.join(os.path.dirname(saved_path) or ".", "_flat_root")
        os.makedirs(root_dir, exist_ok=True)
        super().__init__(root=root_dir, transform=transform, pre_transform=pre_transform)

        add_safe_globals([Data, DataEdgeAttr])
        self.data, self.slices = torch.load(saved_path, weights_only=False)

        if self.verbose:
            try:
                n = self.len()
            except Exception:
                n = "?"
            print(f"[FlatGraphDataset] Loaded {n} graphs from {saved_path}")

    @property
    def raw_file_names(self): return []

    @property
    def processed_file_names(self): return []

    def download(self): pass

    def process(self): pass
