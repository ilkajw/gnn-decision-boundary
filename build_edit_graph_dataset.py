from EditPathGraphDataset import EditPathGraphsDataset, FlatGraphDataset
from torch_geometric.loader import DataLoader
from config import DATASET_NAME, FLIP_AT

if __name__ == "__main__":

    # define input (predicted edit paths sequences and original predictions) paths
    seq_dir = f"data_control/{DATASET_NAME}/predictions/edit_path_graphs_with_predictions_CUMULATIVE_COST"
    base_pred_path = f"data_control/{DATASET_NAME}/predictions/{DATASET_NAME}_predictions.json"

    save_pt = f"data_control/{DATASET_NAME}/processed/{DATASET_NAME}_edit_path_dataset_flip_at_{FLIP_AT}.pt"
    save_meta = f"data_control/{DATASET_NAME}/processed/{DATASET_NAME}_edit_path_dataset_meta_flip_at_{FLIP_AT}.json"

    # build dataset in memory from per-path sequences
    ds = EditPathGraphsDataset(
        seq_dir=seq_dir,
        base_pred_path=base_pred_path,
        flip_at=FLIP_AT/100,
        drop_endpoints=True,
        verbose=True,
    )

    print("In-memory dataset length:", len(ds))

    # save collated dataset and metadata
    ds.save(output_path=save_pt, meta_path=save_meta)
