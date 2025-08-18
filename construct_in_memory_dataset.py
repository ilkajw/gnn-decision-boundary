from EditPathGraphDataset import EditPathGraphsDataset, FlatGraphDataset
from torch_geometric.loader import DataLoader
from config import DATASET_NAME, LABEL_MODE

if __name__ == "__main__":

    # build dataset in memory from per-path sequences
    seq_dir = f"data/{DATASET_NAME}/predictions/edit_path_graphs_with_predictions_CUMULATIVE_COST"
    base_pred_path = f"data/{DATASET_NAME}/predictions/{DATASET_NAME}_predictions.json"

    ds = EditPathGraphsDataset(
        seq_dir=seq_dir,
        base_pred_path=base_pred_path,
        label_mode=LABEL_MODE,
        interpolation="linear",
        k_sigmoid=12.0,
        min_prob=None,
        drop_endpoints=True,
        verbose=True,
        min_flips=True
    )

    print("In-memory dataset length:", len(ds))

    # save collated dataset and metadata
    save_pt = f"data/{DATASET_NAME}/processed/{DATASET_NAME}_edit_path_dataset_{LABEL_MODE}.pt"
    save_meta = f"data/{DATASET_NAME}/processed/{DATASET_NAME}_edit_path_dataset_{LABEL_MODE}_meta.json"
    ds.save(output_path=save_pt, meta_path=save_meta)

    # load and use in DataLoader
    loaded_ds = FlatGraphDataset(saved_path=save_pt, verbose=True)
    loader = DataLoader(loaded_ds, batch_size=32, shuffle=True)
