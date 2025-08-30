import os
# for reproducibility
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

from EditPathGraphDataset import EditPathGraphsDataset
from GAT import GAT
from training_utils import train_epoch, evaluate_loss, evaluate_accuracy, setup_reproducibility
from config import (
    ROOT, DATASET_NAME, K_FOLDS, BATCH_SIZE, EPOCHS, LEARNING_RATE, FLIP_AT  # MODEL_CLS, MODEL_KWARGS
)

# ----- input, output paths ----

# directory with all path sequences (.pt lists)
path_seq_dir = f"data_control/{DATASET_NAME}/pyg_edit_path_graphs"

# path to json with org graph true labels: { "0":{"true_label":0}, "1":{"true_label":1}, ... }
base_labels_path = f"data_control/{DATASET_NAME}/predictions/{DATASET_NAME}_predictions.json"

# output file name definitions
output_dir = f"model_cv_augmented/flip_at_{int(FLIP_AT*100)}/{DATASET_NAME}"
model_fname = f"{DATASET_NAME}_best_model_flip_{int(FLIP_AT*100)}.pt"
split_fname = f"{DATASET_NAME}_best_split_flip_{int(FLIP_AT*100)}.json"
log_fname = f"{DATASET_NAME}_train_log_flip_{int(FLIP_AT*100)}.json"

# run configs
DROP_ENDPOINTS = True
VERBOSE = True

# todo: inject class and delete values below later
HIDDEN_CHANNELS = 8
HEADS = 8
DROPOUT = 0.2


# ----------- helpers -------------
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


# ------ main ------

if __name__ == "__main__":

    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, model_fname)
    split_path = os.path.join(output_dir, split_fname)
    log_path = os.path.join(output_dir, log_fname)

    # for reproducibility
    setup_reproducibility(seed=42)

    # base dataset
    base_ds = TUDataset(
        root=ROOT,
        name=DATASET_NAME,
        transform=Compose([
            to_float_y(),
            drop_edge_attr(),  # remove edge_attr so schema matches path graphs
            tag_origin("org"),
        ])
    )
    labels = hard_labels(base_ds)

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_channels = infer_in_channels(base_ds)
    accuracies = []

    # tracking best across all folds/epochs
    best_acc = -1.0
    best_model_state = None
    best_split = None

    # collect per-fold histories/indices
    fold_records = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(base_ds)), labels), start=1):
        if VERBOSE:
            print(f"\n--- fold {fold} ---")

        # todo: alternatively to current solution,
        #  set DROP_ENDPOINTS=False and train_dataset = path_train

        train_subset = Subset(base_ds, train_idx.tolist())
        test_subset = Subset(base_ds,  test_idx.tolist())

        # augment train subset with path graphs whose endpoints are in train_idx
        allowed_indices = set(map(int, train_idx.tolist()))
        print(f"[info] building augmented dataset...")
        path_train = EditPathGraphsDataset(
            seq_dir=path_seq_dir,
            base_pred_path=base_labels_path,
            flip_at=FLIP_AT,
            drop_endpoints=DROP_ENDPOINTS,
            verbose=False,
            allowed_indices=allowed_indices,  # filter for paths between graphs from train split
        )

        # drop path graph meta data (edit step, cumulative cost, ...) so scheme matches org graphs
        path_train.transform = Compose([
            to_float_y(),
            drop_keys(["edit_step", "cumulative_cost", "source_idx", "target_idx",
                       "iteration", "distance", "num_all_ops"]),
            tag_origin("edit"),
        ])

        print(f"[info] adding {len(path_train)} path graphs to train split...")

        # ---- class distributions (per fold) ----
        base_stats = class_stats(train_subset)
        path_stats = class_stats(path_train)
        # combine by summing counts
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

        # final train = base train + belonging path graphs. test = base test only
        train_dataset = ConcatDataset([train_subset, path_train])
        test_dataset = test_subset

        # for reproducibility
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

        # init model + optimizer
        # todo: inject model class
        model = GAT(
            in_channels=in_channels,
            hidden_channels=HIDDEN_CHANNELS,
            heads=HEADS,
            dropout=DROPOUT,
        ).to(device)

        # model = MODEL_CLS(infer_in_channels(dataset), **MODEL_KWARGS)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # per-fold history
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

        # training epochs
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
                print(f"Epoch {epoch:03d} | train loss: {train_loss:.4f} "
                      f"| test loss: {test_loss:.4f} | test acc: {test_acc:.4f} "
                      f"| time: {int(m):02d}:{s:06.3f}")

            # track best across all folds/epochs
            if test_acc > best_acc:
                if VERBOSE:
                    print(f"New best model at fold {fold}, epoch {epoch} with test acc {test_acc:.4f}")
                best_acc = test_acc
                best_model_state = model.state_dict()
                best_split = {
                    "train_idx": list(map(int, train_idx)),
                    "test_idx":  list(map(int, test_idx)),
                    "fold": fold,
                    "epoch": epoch,
                    "flip_at": FLIP_AT,
                    "drop_endpoints": DROP_ENDPOINTS,
                }

        # end‑of‑fold evaluation
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

    # save best model weights
    torch.save(best_model_state, model_path)

    # save best split info
    with open(split_path, "w") as f:
        json.dump(best_split, f, indent=2)

    #model_config = {
    #    "name": getattr(MODEL_CLS, "__name__", str(MODEL_CLS)),
    #    "kwargs": {k: (v.item() if hasattr(v, "item") else v) for k, v in (MODEL_KWARGS or {}).items()},
    #}

    # consolidated log
    summary = {
        "fold_accuracies": [float(a) for a in accuracies],
        "mean_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
        "std_accuracy": float(np.std(accuracies)) if accuracies else 0.0,
        "best_model": best_split,
        "config": {
            "model": "GAT",  # model_config,
            "HIDDEN_CHANNELS": HIDDEN_CHANNELS,
            "HEADS": HEADS,
            "DROPOUT": DROPOUT,
            "dataset": DATASET_NAME,
            "K_FOLDS": K_FOLDS,
            "EPOCHS": EPOCHS,
            "LEARNING_RATE": LEARNING_RATE,
            "BATCH_SIZE": BATCH_SIZE,
            "stratified": True,
            "augmentation": "train-split path graphs in train split only",
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
        "folds": fold_records
    }
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)

    if VERBOSE:
        print(f"\n Average accuracy over {K_FOLDS} folds: "
              f"{float(np.mean(accuracies)) if accuracies else 0.0:.4f}")
        print(f"Saved best model → {model_path}")
        print(f"Saved log → {log_path}")
