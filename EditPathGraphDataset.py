# editpath_inmemory_dataset.py

import os
import json
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch.serialization import add_safe_globals


class EditPathGraphsDataset(InMemoryDataset):
    """
    Loads all .pt edit-path sequences (each a list[Data]) and flattens them
    into a single InMemoryDataset with a label y per graph.

    Labeling modes:
      - "same_only": keep only graphs from paths whose source and target share the same true label;
                     assign that label to all steps.
      - "source":    assign the source graph's true label to every step.
      - "target":    assign the target graph's true label to every step.
      - "pseudo":    use per-graph model prediction already stored on the Data as `prediction`
                     (optionally filter by probability threshold via `min_prob`).
    """
    # todo: add filtering for certain index sets to build_list

    def __init__(
        self,
        seq_dir,                       # directory with .pt files (each: list[Data])
        base_pred_path,                # JSON with original dataset true labels; keys are string indices
        label_mode="same_only",        # "same_only" | "source" | "target" | "pseudo"
        min_prob=None,                 # for "pseudo": keep only graphs with probability >= min_prob (if provided)
        transform=None,
        pre_transform=None,
        verbose=True,
    ):
        self.seq_dir = seq_dir
        self.base_pred_path = base_pred_path
        self.label_mode = label_mode
        self.min_prob = min_prob
        self.verbose = verbose
        super().__init__(root=None, transform=transform, pre_transform=pre_transform)

        # we build in-memory directly; no disk caching
        data_list = self._build_list()
        self.data, self.slices = self.collate(data_list)
        if self.verbose:
            print(f"[EditPathGraphsDataset] Built dataset with {len(data_list)} graphs "
                  f"from sequences in: {self.seq_dir}")

    # InMemoryDataset expects these
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self):
        pass

    def process(self):
        pass

    # ---------------- internal helpers ----------------

    def _load_base_labels(self):
        with open(self.base_pred_path, "r") as f:
            base = json.load(f)
        # ensure int keys
        return {int(k): int(v["true_label"]) for k, v in base.items()}

    def _label_for_graph(self, g, src_label, tgt_label):
        if self.label_mode == "same_only":
            # only used if caller checks same-class earlier; here we just pick the common label
            assert src_label == tgt_label, "Expected same class in 'same_only' mode."
            return src_label
        elif self.label_mode == "source":
            return src_label
        elif self.label_mode == "target":
            return tgt_label
        elif self.label_mode == "pseudo":
            # use `prediction` on g
            pred = int(getattr(g, "prediction", -1))
            if self.min_prob is not None:
                prob = float(getattr(g, "probability", 0.0))
                if prob < self.min_prob:
                    return None  # skip low-confidence samples
            return pred if pred in (0, 1) else None
        else:
            raise ValueError(f"Unknown label_mode: {self.label_mode}")

    def _build_list(self):
        base_labels = self._load_base_labels()

        add_safe_globals([Data])  # safe load of serialized data
        files = [f for f in os.listdir(self.seq_dir) if f.endswith(".pt")]
        files.sort()

        keep_only_same = (self.label_mode == "same_only")

        data_list = []
        for fname in files:
            path = os.path.join(self.seq_dir, fname)
            seq = torch.load(path, weights_only=False)

            # per-sequence metadata (same for all graphs in seq)
            if not seq:
                continue
            g0 = seq[0]
            i = int(getattr(g0, "source_idx", -1))
            j = int(getattr(g0, "target_idx", -1))

            if i not in base_labels or j not in base_labels:
                if self.verbose:
                    print(f"[WARN] Missing base label for {i} or {j} in {fname}; skipping sequence.")
                continue

            src_label = base_labels[i]
            tgt_label = base_labels[j]

            # optional: keep only same-class sequences for clean supervision
            if keep_only_same and src_label != tgt_label:
                continue

            # add each graph with y
            for g in seq:
                y = self._label_for_graph(g, src_label, tgt_label)
                if y is None:
                    continue  # pseudo with low prob

                # clone minimal fields; attach y
                out = Data(
                    x=g.x,
                    edge_index=g.edge_index,
                    y=torch.tensor([y], dtype=torch.long),
                )

                # (optional) carry over useful metadata
                for key in ("edit_step", "source_idx", "target_idx", "iteration", "distance", "prediction", "probability"):
                    if hasattr(g, key):
                        setattr(out, key, getattr(g, key))

                data_list.append(out)

        return data_list
