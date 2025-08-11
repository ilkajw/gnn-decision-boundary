from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset

from transforms import to_float_y
from EditPathGraphDataset import FlatGraphDataset
from config import DATASET_NAME, ROOT

if __name__ == "__main__":
    # original dataset with cast y -> float, add source=0
    org_ds = TUDataset(root=ROOT, name=DATASET_NAME, transform=to_float_y(domain_flag=0))

    # edit-path dataset (already has y float if used interpolate/same_only ws used)
    edit_pt = f"data/{DATASET_NAME}/processed/{DATASET_NAME}_edit_path_dataset.pt"
    edit_ds = FlatGraphDataset(saved_path=edit_pt, verbose=True)

    # ensure y is float and add source=1
    edit_ds.transform = to_float_y(domain_flag=1)

    # merge
    merged = ConcatDataset([org_ds, edit_ds])

    loader = DataLoader(merged, batch_size=32, shuffle=True)

    # check
    num_mutag = len(org_ds)
    num_edit = len(edit_ds)
    print(f"Merged dataset: {num_mutag} (MUTAG) + {num_edit} (edit-path) = {len(merged)}")