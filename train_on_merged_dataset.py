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
from training_utils import train_epoch, evaluate_accuracy, set_seed
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

    set_seed(42)

    # paths & output files (mirrors original logging structure)
    merged_pt = f"data/{DATASET_NAME}/processed/{DATASET_NAME}_merged_dataset.pt"
    out_dir = f"model_merged/{DATASET_NAME}"
    os.makedirs(out_dir, exist_ok=True)

    model_fname = f"{DATASET_NAME}_model.pt"
    split_fname = "split.json"
    log_fname = "train_log.json"

    model_path = os.path.join(out_dir, model_fname)
    split_path = os.path.join(out_dir, split_fname)
    log_path = os.path.join(out_dir, log_fname)

    # load merged dataset (original + edit-path graphs, collated)
    dataset = FlatGraphDataset(saved_path=merged_pt, verbose=True)

    # folds on hard labels
    labels = hard_labels_for_folds(dataset)
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_channels = infer_in_channels(dataset)

    accuracies = []  # per-fold final test accuracy

    # track best model across all folds/epochs by accuracy
    best_acc = 0.0
    best_model_state = None
    best_split = None

    VERBOSE = False  # match original default (prints only if True)

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(dataset)), labels), start=1):

        if VERBOSE:
            print(f"\n--- fold {fold} ---")

        # split dataset in train/test set
        train_subset = torch.utils.data.Subset(dataset, train_idx.tolist())
        test_subset  = torch.utils.data.Subset(dataset,  test_idx.tolist())

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader  = DataLoader(test_subset,  batch_size=BATCH_SIZE)

        # init model + optimizer
        model = GAT(
            in_channels=in_channels,
            hidden_channels=HIDDEN_CHANNELS,
            heads=HEADS,
            dropout=DROPOUT
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # train model over epochs
        for epoch in range(1, EPOCHS + 1):
            loss = train_epoch(model, train_loader, optimizer, device)
            acc  = evaluate_accuracy(model, test_loader, device)

            if VERBOSE and (epoch % 10 == 0):
                print(f"Epoch {epoch: 03d} | loss: {loss: .4f} | epoch {epoch: 03d} | acc: {acc: .4f}")

            # track best model over folds and epochs (by accuracy)
            if acc > best_acc:
                if VERBOSE:
                    print(f"\n New best is model trained over fold {fold} in epoch {epoch} with acc {acc: .4f}")
                best_acc = acc
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

    # save best model (weights)
    torch.save(best_model_state, model_path)

    # save split info for the best model
    with open(split_path, "w") as f:
        json.dump(best_split, f, indent=2)

    # log k-fold statistics — identical fields to original
    mean_acc = float(np.mean(accuracies)) if len(accuracies) else 0.0
    std_acc  = float(np.std(accuracies)) if len(accuracies) else 0.0
    log = {
        "fold_accuracies": [float(a) for a in accuracies],
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "best_model": best_split
    }
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    if VERBOSE:
        print(f"\n Average accuracy over {K_FOLDS} folds: {mean_acc: .4f}")
        print(f"Saved best model → {model_path}")
        print(f"Saved split → {split_path}")
        print(f"Saved log → {log_path}")
