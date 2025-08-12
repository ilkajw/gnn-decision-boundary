import json
import os
import torch
from pathlib import Path
from torch_geometric.data import InMemoryDataset
from config import DATASET_NAME


def save_dataset_as_inmemory_pt(dataset, save_pt_path: str, meta_path: str | None = None, verbose: bool = True):
    """
    Collate an (iterable) PyG dataset into a single (data, slices) blob and save to .pt.
    Any dataset.transform is applied automatically when iterating.
    """
    # ensure output dir exists
    Path(os.path.dirname(save_pt_path) or ".").mkdir(parents=True, exist_ok=True)
    if meta_path:
        Path(os.path.dirname(meta_path) or ".").mkdir(parents=True, exist_ok=True)

    # collect all Data objects (this will apply transforms)
    data_list = [dataset[i] for i in range(len(dataset))]

    # tiny InMemoryDataset to access `collate`
    class _Collator(InMemoryDataset):
        def __init__(self, data_list):
            # Give a harmless root; silence logs
            super().__init__(root=os.path.join(os.getcwd(), "../_merge_collate_root"), log=False)
            self.data, self.slices = self.collate(data_list)

        @property
        def raw_file_names(self): return []
        @property
        def processed_file_names(self): return []
        def download(self): pass
        def process(self): pass

    collator = _Collator(data_list)
    torch.save((collator.data, collator.slices), save_pt_path)

    if verbose:
        print(f"[merge] Saved merged dataset to {save_pt_path} ({len(data_list)} graphs)")

    if meta_path:
        meta = {
            "num_graphs": len(data_list),
            "sources": getattr(dataset, "sources", None),  # optional
            "dataset_name": DATASET_NAME,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        if verbose:
            print(f"[merge] Saved meta to {meta_path}")