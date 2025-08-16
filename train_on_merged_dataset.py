# train_on_merged_dataset.py — aligned to original training loop/metrics

import os
import json
import random
import numpy as np
import torch

from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold

from EditPathGraphDataset import FlatGraphDataset
from model import GAT
from training_utils import train_epoch, evaluate_accuracy, set_seed, evaluate_loss
from config import (
    DATASET_NAME, EPOCHS, LEARNING_RATE, BATCH_SIZE,
    HIDDEN_CHANNELS, HEADS, DROPOUT, K_FOLDS
)


# ----------------------- helpers -----------------------

def infer_in_channels(dataset):
    sample = dataset[0]
    return sample.x.size(-1)


def hard_labels_for_folds(dataset):
    """Hard labels (0/1) for StratifiedKFold; robust to soft targets."""
    ys = []
    for i in range(len(dataset)):
        yi = dataset[i].y
        if isinstance(yi, torch.Tensor):
            yv = float(yi.view(-1)[0])
        else:
            yv = float(yi)
        ys.append(int(yv > 0.5))
    return ys


# ---------------------------------- run training ----------------------------------

if __name__ == "__main__":

    VERBOSE = True

    model_fname = f"{DATASET_NAME}_model.pt"
    split_fname = "split.json"
    log_fname = "train_log.json"

    set_seed(42)

    # paths & output files
    merged_pt = f"data/{DATASET_NAME}/processed/{DATASET_NAME}_merged_dataset.pt"
    out_dir = f"model_merged/{DATASET_NAME}"
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, model_fname)
    split_path = os.path.join(out_dir, split_fname)  # todo: save split path? currently not
    log_path = os.path.join(out_dir, log_fname)

    # load merged dataset (original + edit-path graphs, collated)
    dataset = FlatGraphDataset(saved_path=merged_pt, verbose=True)

    # folds on hard labels
    labels = hard_labels_for_folds(dataset)
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_channels = infer_in_channels(dataset)

    # to track best model across all folds/epochs by accuracy
    accuracies = []  # per-fold final test accuracy
    best_acc = -1.0
    best_model_state = None
    best_split = None

    # collect all folds' info
    fold_records = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(dataset)), labels), start=1):

        if VERBOSE:
            print(f"\n--- fold {fold} ---")

        # split dataset in train/test set
        train_subset = torch.utils.data.Subset(dataset, train_idx.tolist())
        test_subset = torch.utils.data.Subset(dataset,  test_idx.tolist())

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_subset,  batch_size=BATCH_SIZE)

        # init model + optimizer
        model = GAT(
            in_channels=in_channels,
            hidden_channels=HIDDEN_CHANNELS,
            heads=HEADS,
            dropout=DROPOUT
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # per-fold history
        hist = {"train_loss": [], "test_loss": [], "test_acc": [], "num_epochs": EPOCHS}

        # train model over epochs
        for epoch in range(1, EPOCHS + 1):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            test_acc = evaluate_accuracy(model, test_loader, device)
            test_loss = evaluate_loss(model, test_loader, device)

            # record curves
            hist["train_loss"].append(float(train_loss))
            hist["test_loss"].append(float(test_loss))
            hist["test_acc"].append(float(test_acc))

            if VERBOSE and (epoch % 10 == 0):
                print(f"Epoch {epoch: 03d} | train loss: {train_loss: .4f} | test loss: {test_loss: .4f} | test acc: {test_acc: .4f}")

            # track best model over folds and epochs by accuracy
            if test_acc > best_acc:
                if VERBOSE:
                    print(f"\n New best is model trained over fold {fold} in epoch {epoch} with acc {test_acc: .4f}")
                best_acc = test_acc
                best_model_state = model.state_dict()
                best_split = {
                    'train_idx': train_idx.tolist(),
                    'test_idx':  test_idx.tolist(),
                    'fold': fold,
                    'epoch': epoch
                }

        # evaluate model trained over full fold
        final_acc = evaluate_accuracy(model, test_loader, device)
        accuracies.append(final_acc)
        if VERBOSE:
            print(f"Fold {fold} | Accuracy: {final_acc: .4f}")

        # stash everything for this fold
        fold_records.append({
            "fold": fold,
            "indices": {"train_idx": list(map(int, train_idx)),
                        "test_idx": list(map(int, test_idx))},
            "history": hist,
            "final_accuracy": float(final_acc),
        })

    # save best model (weights)
    torch.save(best_model_state, model_path)

    # save split info for the best model
    with open(split_path, "w") as f:
        json.dump(best_split, f, indent=2)

    # consolidated log with summary + all fold records
    summary = {
        "fold_accuracies": [float(a) for a in accuracies],
        "mean_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
        "std_accuracy":  float(np.std(accuracies)) if accuracies else 0.0,
        "best_model": best_split,
        "config": {
            "K_FOLDS": K_FOLDS,
            "EPOCHS": EPOCHS,
            "LEARNING_RATE": LEARNING_RATE,
            "BATCH_SIZE": BATCH_SIZE,
            "HIDDEN_CHANNELS": HIDDEN_CHANNELS,
            "HEADS": HEADS,
            "DROPOUT": DROPOUT,
            "device": str(device),
            "dataset": DATASET_NAME,
            "stratified": True,
        },
        "folds": fold_records
    }
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)

    if VERBOSE:
        print(f"\n Average accuracy over {K_FOLDS} folds: {float(np.mean(accuracies)) if accuracies else 0.0: .4f}")
        print(f"Saved best model → {model_path}")
        print(f"Saved log → {log_path}")
