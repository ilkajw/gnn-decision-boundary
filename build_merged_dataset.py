from torch.utils.data import ConcatDataset
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset

from EditPathGraphDataset import FlatGraphDataset
from config import DATASET_NAME, ROOT
from dataset_utils import save_dataset_as_inmemory_pt
from transforms import to_float_y

if __name__ == "__main__":

    # load original dataset with cast y->float, tag domain=0
    org_ds = TUDataset(root=ROOT, name=DATASET_NAME, transform=to_float_y(domain_flag=0))

    # load previously build edit-path dataset
    edit_pt = f"data/{DATASET_NAME}/processed/{DATASET_NAME}_edit_path_dataset.pt"
    edit_ds = FlatGraphDataset(saved_path=edit_pt, verbose=True)
    edit_ds.transform = to_float_y(domain_flag=1)  # ensure y is float + tag domain=1 when accessed

    # merge original dataset and edit-path dataset
    merged = ConcatDataset([org_ds, edit_ds])

    # check
    loader = DataLoader(merged, batch_size=32, shuffle=True)
    num_mutag = len(org_ds)
    num_edit = len(edit_ds)
    print(f"Merged dataset: {num_mutag} (MUTAG) + {num_edit} (edit-path) = {len(merged)}")

    # save merged dataset as a single (data, slices) .pt

    merged_pt = f"data/{DATASET_NAME}/processed/{DATASET_NAME}_merged_dataset.pt"
    merged_meta = f"data/{DATASET_NAME}/processed/{DATASET_NAME}_merged_dataset_meta.json"

    save_dataset_as_inmemory_pt(merged, merged_pt, meta_path=merged_meta, verbose=True)