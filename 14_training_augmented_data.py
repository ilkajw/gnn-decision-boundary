import os
# For reproducibility
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "42"

import json
import numpy as np
import torch
from time import perf_counter
from sklearn.model_selection import StratifiedKFold
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset, Subset
from torch_geometric.transforms import Compose

from EditPathGraphDataset import EditPathGraphDataset
from training_utils import train_epoch, evaluate_loss, evaluate_accuracy, setup_reproducibility
from config import (
    ROOT, DATASET_NAME, K_FOLDS, BATCH_SIZE, EPOCHS, LEARNING_RATE, FLIP_AT, MODEL,
    PREDICTIONS_DIR, MODEL_CLS, MODEL_KWARGS
)

# todo: alternatively to current solution,
#  set DROP_ENDPOINTS=False and train_dataset = path_train

# todo: change true label source from dict to actual dataset

# ----- Set input, output paths ----
# todo: adjust the paths correctly after GAT training
# Directory with all path sequences (.pt lists)
# using hard coded path to not have to run predictions with GCN as we only need the true label
path_seq_dir = "data_actual_best/MUTAG/GAT/predictions/edit_path_graphs_with_predictions_CUMULATIVE_COST"
# f"{PREDICTIONS_DIR}/edit_path_graphs_with_predictions_CUMULATIVE_COST"

# Path to json with org graph true labels: { "0":{"true_label":0}, "1":{"true_label":1}, ... }
base_labels_path = "data_actual_best/MUTAG/GAT/predictions/MUTAG_GAT_predictions.json"
# f"{PREDICTIONS_DIR}/{DATASET_NAME}_{MODEL}_predictions.json"

# Output files definition
output_dir = f"model_cv_augmented/{DATASET_NAME}/{MODEL}/flip_at_{int(FLIP_AT*100)}"
# todo: later back to: f"models_cv_augmented/{DATASET_NAME}/{MODEL}/{flip_at_{int(FLIP_AT*100)}/"
model_fname = f"{DATASET_NAME}_{MODEL}_best_model_flip_{int(FLIP_AT*100)}.pt"
split_fname = f"{DATASET_NAME}_{MODEL}_best_split_flip_{int(FLIP_AT*100)}.json"
log_fname = f"{DATASET_NAME}_{MODEL}_train_log_flip_{int(FLIP_AT*100)}.json"

# Set run parameters
DROP_ENDPOINTS = True  # drop source and target graphs in EditPathGraphDataset
VERBOSE = True  # print training progress


# ---- Helpers ----
def infer_in_channels(dataset):
    sample = dataset[0]
    return sample.x.size(-1)


def hard_labels(ds):
    """Hard labels (0/1) for StratifiedKFold; robust to float/soft targets."""
    ys = []
    for i in range(len(ds)):
        yi = ds[i].y
        yv = float(yi.view(-1)[0]) if isinstance(yi, torch.Tensor) else float(yi)
        ys.append(int(yv > 0.5))
    return ys


def to_float_y():
    def _tf(data):
        y = data.y
        if not torch.is_floating_point(y):
            y = y.float()
        if y.dim() == 0:
            y = y.unsqueeze(0)
        data.y = y
        return data
    return _tf


def drop_edge_attr():
    def _tf(data):
        if 'edge_attr' in data:
            del data.edge_attr
        return data
    return _tf


def drop_keys(keys):
    keys = set(keys)

    def _tf(data):
        for k in list(data.keys()):
            if k in keys:
                delattr(data, k)
        return data
    return _tf


def tag_origin(tag: str):
    def _tf(data):
        data.origin = tag
        data.is_original = 1 if tag == "org" else 0
        return data
    return _tf


def class_stats(dataset, batch_size=2048):
    """
    Returns dict with counts and proportions of class 0/1 in a (sub)dataset.
    Assumes data.y is float in {0.0, 1.0} shaped [1] per sample.
    """
    if len(dataset) == 0:
        return {"n": 0, "n0": 0, "n1": 0, "p0": 0.0, "p1": 0.0}

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    n0 = n1 = 0
    for batch in loader:
        y = batch.y.view(-1)
        n1 += int((y > 0.5).sum().item())
        n0 += int(y.numel() - (y > 0.5).sum().item())
    n = n0 + n1
    p0 = (n0 / n) if n > 0 else 0.0
    p1 = (n1 / n) if n > 0 else 0.0
    return {"n": n, "n0": n0, "n1": n1, "p0": p0, "p1": p1}


# ------ Run ------

if __name__ == "__main__":

    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, model_fname)
    split_path = os.path.join(output_dir, split_fname)
    log_path = os.path.join(output_dir, log_fname)

    # for reproducibility
    setup_reproducibility(seed=42)

    base_ds = TUDataset(
        root=ROOT,
        name=DATASET_NAME,
        transform=Compose([
            to_float_y(),  # Ensure float labels
            drop_edge_attr(),  # Remove edge_attr to match path graph schema
            tag_origin("org"),  # Tag each graph origin "original"
        ])
    )

    labels = hard_labels(base_ds)  # In case of soft labelling, derive hard labels for StratifiedKFold

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_channels = infer_in_channels(base_ds)
    accuracies = []

    # To track best model
    best_acc = -1.0
    best_model_state = None
    best_split = None

    # To collect per-fold info
    fold_records = []

    if VERBOSE:
        print(f"--- Training {MODEL} model on augmented data with flips at {FLIP_AT} ---")

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(base_ds)), labels), start=1):

        if VERBOSE:
            print(f"\n--- fold {fold} ---")

        # Define train and test split on original dataset
        train_subset = Subset(base_ds, train_idx.tolist())
        test_subset = Subset(base_ds,  test_idx.tolist())

        # Augment train subset with path graphs between training graphs
        allowed_indices = set(map(int, train_idx.tolist()))
        print(f"[info] Building augmented dataset...")
        path_train = EditPathGraphDataset(
            seq_dir=path_seq_dir,
            base_pred_path=base_labels_path,
            flip_at=FLIP_AT,
            drop_endpoints=DROP_ENDPOINTS,
            verbose=False,
            allowed_indices=allowed_indices,  # Filter for paths between graphs from train split
            use_base_dataset=False,  # todo: change to True and base_ds later
            base_dataset=None
        )

        path_train.transform = Compose([
            to_float_y(),  # Ensure float labels

            drop_keys(["edit_step", "cumulative_cost", "source_idx",  # Drop attrs to match org schema for collating
                       "target_idx", "iteration", "distance",
                       "num_all_ops", "prediction", "probability"]),
            tag_origin("edit"),  # Tag each graph with origin "edit path"
        ])

        print(f"[info] Adding {len(path_train)} path graphs to train split...")

        # ---- Class distributions per fold ----
        base_stats = class_stats(train_subset)
        path_stats = class_stats(path_train)
        train_n0 = base_stats["n0"] + path_stats["n0"]
        train_n1 = base_stats["n1"] + path_stats["n1"]
        train_n = train_n0 + train_n1
        train_stats = {
            "n": train_n,
            "n0": train_n0,
            "n1": train_n1,
            "p0": (train_n0 / train_n) if train_n > 0 else 0.0,
            "p1": (train_n1 / train_n) if train_n > 0 else 0.0,
        }

        if VERBOSE:
            print(f"[info] Path graph classes: 0: {path_stats['n0']}, 1: {path_stats['n1']}")

        # Final train set = base train + belonging path graphs,
        # final test set = base test
        train_dataset = ConcatDataset([train_subset, path_train])
        test_dataset = test_subset

        # For reproducibility
        g = torch.Generator()
        g.manual_seed(42)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            persistent_workers=False,
            generator=g
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            persistent_workers=False
        )

        # Init model + optimizer
        model = MODEL_CLS(infer_in_channels(base_ds), **MODEL_KWARGS).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Per-fold history
        hist = {"train_loss": [], "test_loss": [], "test_acc": [], "num_epochs": EPOCHS,
                "sizes": {
                    "base_train": len(train_subset),
                    "path_train": len(path_train),
                    "train_total": len(train_subset) + len(path_train),
                    "test": len(test_subset),
                    "class_distribution": {
                        "base_train": base_stats,  # {n,n0,n1,p0,p1}
                        "path_train": path_stats,  # {n,n0,n1,p0,p1}
                        "train_total": train_stats  # {n,n0,n1,p0,p1}
                    }
                }}

        # Training epochs
        for epoch in range(1, EPOCHS + 1):
            epoch_t0 = perf_counter()

            train_loss = train_epoch(model, train_loader, optimizer, device)
            test_acc = evaluate_accuracy(model, test_loader, device)
            test_loss = evaluate_loss(model, test_loader, device)

            epoch_sec = perf_counter() - epoch_t0
            m, s = divmod(epoch_sec, 60)

            hist["train_loss"].append(float(train_loss))
            hist["test_loss"].append(float(test_loss))
            hist["test_acc"].append(float(test_acc))

            if VERBOSE:
                print(f"Epoch {epoch: 03d} | train loss: {train_loss: .4f} "
                      f"| test loss: {test_loss: .4f} | test acc: {test_acc: .4f} "
                      f"| time: {int(m): 02d}:{s: 06.3f}")

            # Track best across all folds/epochs
            if test_acc > best_acc:
                if VERBOSE:
                    print(f"[info] New best model in fold {fold}, epoch {epoch} with test acc {test_acc: .4f}")
                best_acc = test_acc
                best_model_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
                best_split = {
                    "train_idx": list(map(int, train_idx)),
                    "test_idx":  list(map(int, test_idx)),
                    "fold": fold,
                    "epoch": epoch,
                    "flip_at": FLIP_AT,
                    "drop_endpoints": DROP_ENDPOINTS,
                }

        # End‑of‑fold evaluation
        final_acc = evaluate_accuracy(model, test_loader, device)
        accuracies.append(final_acc)
        if VERBOSE:
            print(f"Fold {fold} | Accuracy: {final_acc:.4f} | "
                  f"train_total={len(train_subset)+len(path_train)} "
                  f"(base {len(train_subset)} + path {len(path_train)})")

        fold_records.append({
            "fold": fold,
            "indices": {
                "train_idx": list(map(int, train_idx)),
                "test_idx":  list(map(int, test_idx))
            },
            "history": hist,
            "final_accuracy": float(final_acc),
        })

    # Save
    torch.save(best_model_state, model_path)

    with open(split_path, "w") as f:
        json.dump(best_split, f, indent=2)

    # Consolidated log
    model_config = {
        "name": getattr(MODEL_CLS, "__name__", str(MODEL_CLS)),
        "kwargs": {k: (v.item() if hasattr(v, "item") else v) for k, v in (MODEL_KWARGS or {}).items()},
    }

    log = {
        "fold_test_accuracies": [float(a) for a in accuracies],
        "mean_test_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
        "std_accuracy": float(np.std(accuracies)) if accuracies else 0.0,
        "config": {
            "dataset": DATASET_NAME,
            "model": model_config,
            "K_FOLDS": K_FOLDS,
            "EPOCHS": EPOCHS,
            "LEARNING_RATE": LEARNING_RATE,
            "BATCH_SIZE": BATCH_SIZE,
            "stratified": True,
            "augmentation": "Train-split path graphs in train split",
            "flip_at": FLIP_AT,
            "drop_endpoints": DROP_ENDPOINTS if "DROP_ENDPOINTS" in globals() else None,
            "env": {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda": getattr(torch.version, "cuda", None),
                "cudnn": torch.backends.cudnn.version(),
                "device": str(device),
            },
        },
        "best_split_info": best_split,
        "folds": fold_records
    }
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    if VERBOSE:
        print(f"\n [info] Average test accuracy over {K_FOLDS} folds: "
              f"{float(np.mean(accuracies)) if accuracies else 0.0: .4f}")
        print(f"[info] Saved best model → {model_path}")
        print(f"[info] Saved log → {log_path}")
