from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import Compose

from EditPathGraphDataset import FlatGraphDataset
from config import DATASET_NAME, ROOT, LABEL_MODE
from dataset_utils import save_dataset_as_inmemory_pt
from data_transform_utils import to_float_y, drop_edge_attr, tag_origin

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
    edit_pt = f"data/{DATASET_NAME}/processed/{DATASET_NAME}_edit_path_dataset_{LABEL_MODE}.pt"
    edit_ds = FlatGraphDataset(saved_path=edit_pt, verbose=True)
    edit_ds.transform = Compose([
        to_float_y(),
        tag_origin("edit")  # tag as "edit"
    ])

    # merge original dataset and edit-path dataset
    merged = ConcatDataset([org_ds, edit_ds])

    # check with loader
    loader = DataLoader(merged, batch_size=32, shuffle=True)
    num_org = len(org_ds)
    num_edit = len(edit_ds)
    print(f"Merged dataset: {num_org} ({DATASET_NAME}) + {num_edit} (edit-path) = {len(merged)}")

    # save merged dataset as a single (data, slices) .pt
    merged_pt = f"data/{DATASET_NAME}/processed/{DATASET_NAME}_merged_dataset_tagged.pt"
    merged_meta = f"data/{DATASET_NAME}/processed/{DATASET_NAME}_merged_dataset_meta_tagged.json"

    save_dataset_as_inmemory_pt(merged, merged_pt, meta_path=merged_meta, verbose=True)
