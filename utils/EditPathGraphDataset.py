import os
import json
import torch

from pathlib import Path
from torch_geometric.data import InMemoryDataset, Data
from torch.serialization import add_safe_globals

from config import DATASET_NAME, DISTANCE_MODE, ROOT


def _ensure_dir(path):
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)


class EditPathGraphsDataset(InMemoryDataset):
    """
    In-memory dataset from per-path .pt sequences (lists of PyG Data).

    Labeling strategy:
      • If source and target have the same class: assign that class to all graphs on the path.
      • If source and target graph classes differ: assign source class up to progress t < flip_at,
        then target class for t >= flip_at.
        - t is the normalized progress along the edit path w.r.t. config.DISTANCE_MODE:
          * 'cost'    -> t = cumulative_cost / distance
          * 'num_ops' -> t = edit_step / num_all_ops

    Parameters
    ----------
    seq_dir : str
        Directory with per-path .pt sequences (each a list[PyG Data]).
    base_pred_path : str
        JSON with true labels of base graphs, indexed by graph id.
        Expected shape: { "<idx>": {"true_label": 0 or 1}, ... }
    flip_at : float, default 0.5
        Threshold in [0,1] where the label flips for different-class paths.
        For t < flip_at -> source label; for t >= flip_at -> target label.
    drop_endpoints : bool, default True
        If True, drop source and target graphs in each sequence.
    verbose : bool, default True
        Print info/warnings.
    """

    def __init__(
        self,
        seq_dir,
        base_pred_path,
        flip_at: float = 0.5,
        transform=None,
        pre_transform=None,
        drop_endpoints: bool = True,
        verbose: bool = True,
        allowed_indices: set[int] | None = None,
    ):
        self.seq_dir = seq_dir
        self.base_pred_path = base_pred_path
        self.drop_endpoints = bool(drop_endpoints)
        self.verbose = bool(verbose)
        self.allowed_indices = set(allowed_indices) if allowed_indices is not None else None
        self.flip_at = float(flip_at)

        assert 0.0 <= self.flip_at <= 1.0, "flip_at must be in [0,1]"

        root_dir = os.path.abspath(f"{ROOT}/{DATASET_NAME}/processed/_editpath_inmem_root")
        # was for last hopefully correct run:
        # root_dir = os.path.abspath(f"data_control/{DATASET_NAME}/processed/_editpath_inmem_root")

        super().__init__(root=root_dir, transform=transform, pre_transform=pre_transform)

        data_list = self._build_list()
        self.data, self.slices = self.collate(data_list)
        if self.verbose:
            print(f"[EditPathGraphsDataset] {len(data_list)} graphs from {self.seq_dir} | flip_at={self.flip_at}")

    # InMemoryDataset expects these
    @property
    def raw_file_names(self): return []

    @property
    def processed_file_names(self): return []

    def download(self): pass
    def process(self): pass

    # -------------- internal helpers -------------------------

    def _load_base_labels(self):
        """
        True classes of original dataset graphs.
        """
        with open(self.base_pred_path, "r") as f:
            base = json.load(f)
        # ensure int keys
        return {int(k): int(v["true_label"]) for k, v in base.items()}

    def _t_from_graph(self, g):
        """
        Normalized progress t in [0,1] based on config.DISTANCE_MODE.
        """
        if DISTANCE_MODE == "cost":
            dist = float(getattr(g, "distance"))
            step = float(getattr(g, "cumulative_cost"))
        elif DISTANCE_MODE == "edit_step":
            dist = float(getattr(g, "num_all_ops"))
            step = float(getattr(g, "edit_step"))
        else:
            print(f"[warn] config.DISTANCE_MODE has unexpected value '{DISTANCE_MODE}'. "
                  f"Expected 'cost' or 'edit_step'. Defaulting to 'cost'.")
            dist = float(getattr(g, "distance"))
            step = float(getattr(g, "cumulative_cost"))
        t = max(0.0, min(step / dist, 1.0)) if dist > 0 else 0.0

        # To test correctness
        # i = getattr(g, "source_idx")
        # j = getattr(g, "target_idx")
        # print(f"{i}, {j}: t: {t}")

        return t

    def _label_for_graph(self, g, y_src: int, y_tgt: int):
        """
        Hard label following the flip-at-threshold policy.
        """
        if y_src == y_tgt:
            return float(y_src)
        t = self._t_from_graph(g)
        return float(y_src if t < self.flip_at else y_tgt)

    def _build_list(self):

        add_safe_globals([Data])

        base_labels = self._load_base_labels()
        files = sorted([f for f in os.listdir(self.seq_dir) if f.endswith(".pt")])

        data_list = []
        no_intermediates = []

        if not os.path.isdir(self.seq_dir):
            if self.verbose:
                print(f"[WARN] seq_dir '{self.seq_dir}' does not exist.")

        # Loop through sequence files, then through sequence to add graphs
        for fname in files:

            seq = torch.load(os.path.join(self.seq_dir, fname), weights_only=False)
            if not seq:
                if self.verbose:
                    print(f"[WARN] missing graph sequence expected in {fname}")
                continue

            g0 = seq[0]
            i = int(getattr(g0, "source_idx", -1))
            j = int(getattr(g0, "target_idx", -1))

            # Filter for paths between graphs from training split
            if self.allowed_indices is not None:
                if (i not in self.allowed_indices) or (j not in self.allowed_indices):
                    continue

            if i not in base_labels or j not in base_labels:
                if self.verbose:
                    print(f"[WARN] missing base label for {i} or {j} in {fname}.")
                continue
            y_src = base_labels[i]
            y_tgt = base_labels[j]

            # Optionally drop source and target graph
            if self.drop_endpoints:
                if len(seq) <= 2:
                    no_intermediates.append((i, j))
                    if self.verbose:
                        print(f"[INFO] {fname}: dropped endpoints and found no intermediate graphs. skipping path.")
                    continue
                seq_iter = seq[1:-1]
            else:
                seq_iter = seq

            # Add each path graph with label and attributes to dataset
            for g in seq_iter:
                y_val = self._label_for_graph(g, y_src, y_tgt)

                data_point = Data(
                    x=g.x,
                    edge_index=g.edge_index,
                    y=torch.tensor([y_val], dtype=torch.float),
                )

                for key in (
                    "edit_step", "cumulative_cost", "source_idx", "target_idx", "iteration",
                    "distance", "prediction", "probability", "num_all_ops"
                ):
                    if hasattr(g, key):
                        setattr(data_point, key, getattr(g, key))
                    else:
                        if self.verbose:
                            print(f"[WARN] Missing attribute {key} at edit step "
                                  f"{getattr(g, 'edit_step', '?')} between {i}, {j}.")

                data_list.append(data_point)

        # For testing purposes only
        os.makedirs(f"{ROOT}/{DATASET_NAME}/test/", exist_ok=True)
        with open(
            f"{ROOT}/{DATASET_NAME}/test/"
            f"{DATASET_NAME}_no_intermediate_graphs_at_dataset_build.json", "w"
        ) as f:
            json.dump(no_intermediates, f, indent=2)

        return data_list

    def save(self, dataset_output_path, meta_output_path=None):
        """
        Save the collated PyG dataset (data + slices) to a single .pt file.
        Optionally save JSON meta for reproducibility.
        """
        _ensure_dir(dataset_output_path)
        torch.save((self.data, self.slices), dataset_output_path)

        if meta_output_path:
            _ensure_dir(meta_output_path)
            meta = {
                "seq_dir": self.seq_dir,
                "base_pred_path": self.base_pred_path,
                "flip_at": self.flip_at,
                "distance_mode": DISTANCE_MODE,
                "num_graphs": self.len(),
            }
            with open(meta_output_path, "w") as f:
                json.dump(meta, f, indent=2)

        if self.verbose:
            print(f"[EditPathGraphsDataset] Saved dataset to {dataset_output_path}"
                  f"{' and meta to ' + meta_output_path if meta_output_path else ''}")
