import json
import os
import torch
from pathlib import Path
from torch.utils.data import ConcatDataset
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import Compose

from EditPathGraphDataset import FlatGraphDataset
from config import DATASET_NAME, ROOT, LABEL_MODE


# ----- helpers ---------

def to_float_y():
    """
    Returns a transform that casts y -> float tensor [0,1].
    """
    def _tf(data):
        y = data.y
        if not torch.is_floating_point(y):
            y = y.float()
        if y.dim() == 0:  # make sure it's at least shape [1]
            y = y.unsqueeze(0)
        data.y = y
        return data
    return _tf


def drop_edge_attr():
    def _tf(data):
        if 'edge_attr' in data:
            del data.edge_attr  # remove attribute from the Data object
        return data
    return _tf


def tag_origin(tag: str):
    assert tag in ("org", "edit")
    def _apply(data):
        data.origin = tag           # "org" or "edit"
        data.is_original = 1 if tag == "org" else 0
        return data
    return _apply


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

    # InMemoryDataset to access `collate`
    class _Collator(InMemoryDataset):
        def __init__(self, data_list):
            # Give a harmless root; silence logs
            super().__init__(root=os.path.join(os.getcwd(), "_merge_collate_root"), log=False)
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


# ------ build dataset --------

if __name__ == "__main__":

    # load original dataset with cast y->float
    org_ds = TUDataset(
        root=ROOT,
        name=DATASET_NAME,
        transform=Compose([
            to_float_y(),
            drop_edge_attr(),
            tag_origin("org")  # tag as "org"
        ])
    )

    # load previously build edit-path dataset
    edit_pt = f"data_control/{DATASET_NAME}/processed/{DATASET_NAME}_edit_path_dataset_{LABEL_MODE}.pt"
    edit_ds = FlatGraphDataset(saved_path=edit_pt, verbose=True)
    edit_ds.transform = Compose([
        to_float_y(),
        tag_origin("edit")  # tag as "edit"
    ])

    # merge original dataset and edit-path dataset
    merged = ConcatDataset([org_ds, edit_ds])

    # load just to check
    loader = DataLoader(merged, batch_size=32, shuffle=True)
    num_org = len(org_ds)
    num_edit = len(edit_ds)
    print(f"Merged dataset: {num_org} ({DATASET_NAME}) + {num_edit} (edit-path) = {len(merged)}")

    # save merged dataset as a single (data, slices) .pt
    merged_pt = f"data_control/{DATASET_NAME}/processed/{DATASET_NAME}_merged_dataset_tagged.pt"
    merged_meta = f"data_control/{DATASET_NAME}/processed/{DATASET_NAME}_merged_dataset_meta_tagged.json"

    save_dataset_as_inmemory_pt(merged, merged_pt, meta_path=merged_meta, verbose=True)
