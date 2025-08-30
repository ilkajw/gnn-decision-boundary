# train_on_merged_dataset.py — aligned to original training loop/metrics
import json
import numpy as np
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":16:8"
os.environ["PYTHONHASHSEED"] = "42"

import torch
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_num_threads(1)

from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from time import perf_counter

from EditPathGraphDataset import FlatGraphDataset
from GAT import GAT
from training_utils import train_epoch, evaluate_accuracy, set_seed, evaluate_loss
from config import (
    DATASET_NAME, EPOCHS, LEARNING_RATE, BATCH_SIZE,
    HIDDEN_CHANNELS, HEADS, DROPOUT, K_FOLDS, FLIP_AT
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

def _fmt_optional(metric):
    """Format optional float for printing."""
    return "N/A" if metric is None else f"{metric: .4f}"

def detect_origin_indices(dataset):
    """
    Return (original_idx, extra_idx) using data.origin.
    origin must be 'org' (original MUTAG) or 'edit' (synthetic).
    """
    original_idx, extra_idx = [], []
    for i, g in enumerate(dataset):
        if not hasattr(g, "origin"):
            raise RuntimeError(f"Data object at index {i} has no .origin field")
        val = str(g.origin).lower()
        if val == "org":
            original_idx.append(i)
        elif val == "edit":
            extra_idx.append(i)
        else:
            raise RuntimeError(
                f"Data object at index {i} has invalid origin='{g.origin}' (expected 'org' or 'edit')"
            )
    return original_idx, extra_idx

# ---------------------------------- run training ----------------------------------

if __name__ == "__main__":

    VERBOSE = True

    model_fname = f"{DATASET_NAME}_model_merged_plus_org_eval.pt"
    split_fname = "train_test_split_model_merged_plus_org_eval.json"
    log_fname = "train_log_model_merged_plus_org_eval.json"

    set_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # paths & output files
    merged_pt = f"data_control/{DATASET_NAME}/processed/{DATASET_NAME}_merged_dataset_tagged.pt"
    out_dir = f"model_merged_plus_org_eval_control/{DATASET_NAME}"
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, model_fname)
    split_path = os.path.join(out_dir, split_fname)
    log_path = os.path.join(out_dir, log_fname)

    # load merged dataset (original + edit-path graphs, collated)
    dataset = FlatGraphDataset(saved_path=merged_pt, verbose=True)

    # identify which indices are original MUTAG
    org_idx, edit_idx = detect_origin_indices(dataset)

    assert len(org_idx) > 0, "No 'org' samples found. Check tagging in the merged dataset."
    assert len(edit_idx) > 0, "No 'edit' samples found. Check tagging in the merged dataset."
    if VERBOSE:
        print(f"Detected {len(org_idx)} original graphs and {len(edit_idx)} edit graphs.")

    # folds on hard labels
    labels = hard_labels_for_folds(dataset)
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_channels = infer_in_channels(dataset)

    # to track best model across all folds/epochs by accuracy
    accuracies = []  # per-fold final test accuracy
    mutag_accuracies = []  # per-fold final MUTAG-only accuracy
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

        g = torch.Generator()
        g.manual_seed(42)
        train_loader = DataLoader(train_subset,
                                  batch_size=BATCH_SIZE,
                                  num_workers=0,
                                  persistent_workers=False,
                                  shuffle=True,
                                  generator=g
                                  )
        test_loader = DataLoader(test_subset,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False,
                                 num_workers=0,
                                 persistent_workers=False,
                                 )

        # define MUTAG-only view of test split to evaluate original accuracy
        test_indices_list = test_idx.tolist()
        mutag_test_indices = [i for i in test_indices_list if i in org_idx]
        if len(mutag_test_indices) > 0:
            mutag_test_subset = torch.utils.data.Subset(dataset, mutag_test_indices)
            mutag_test_loader = DataLoader(
                mutag_test_subset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                persistent_workers=False,
            )
        else:
            mutag_test_loader = None

        # init model + optimizer
        model = GAT(
            in_channels=in_channels,
            hidden_channels=HIDDEN_CHANNELS,
            heads=HEADS,
            dropout=DROPOUT
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # per-fold history
        hist = {"train_loss": [],
                "test_loss": [],
                "test_acc": [],
                "mutag_test_loss": [],
                "mutag_test_acc": [],
                "num_epochs": EPOCHS}

        # train model over epochs
        for epoch in range(1, EPOCHS + 1):
            epoch_t0 = perf_counter()

            train_loss = train_epoch(model, train_loader, optimizer, device)
            test_acc = evaluate_accuracy(model, test_loader, device)
            test_loss = evaluate_loss(model, test_loader, device)

            # evaluate on original test graphs only
            if mutag_test_loader is not None:
                mutag_acc = evaluate_accuracy(model, mutag_test_loader, device)
                mutag_loss = evaluate_loss(model, mutag_test_loader, device)
                hist["mutag_test_acc"].append(float(mutag_acc))
                hist["mutag_test_loss"].append(float(mutag_loss))
            else:
                mutag_acc, mutag_loss = None, None
                hist["mutag_test_acc"].append(None)
                hist["mutag_test_loss"].append(None)

            epoch_sec = perf_counter() - epoch_t0  # whole epoch duration (train + eval)
            m, s = divmod(epoch_sec, 60)

            # record
            hist["train_loss"].append(float(train_loss))
            hist["test_loss"].append(float(test_loss))
            hist["test_acc"].append(float(test_acc))

            if VERBOSE:
                print(f"Epoch {epoch: 03d} | train loss: {train_loss: .4f} | test loss: {test_loss: .4f} | "
                      f"test acc: {test_acc: .4f} | {DATASET_NAME} valid acc: {_fmt_optional(mutag_acc)} |"
                      f"time: {int(m): 02d}:{s: 06.3f}")

            # track best model over folds and epochs by accuracy
            if round(float(test_acc), 6) > round(float(best_acc), 6):
                if VERBOSE:
                    print(f"New best is model trained over fold {fold} in epoch {epoch} with test acc {test_acc: .4f}")
                best_acc = float(test_acc)
                best_model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
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

        # evaluate final org test graph accuracy
        if mutag_test_loader is not None:
            final_mutag_acc = evaluate_accuracy(model, mutag_test_loader, device)
        else:
            final_mutag_acc = None
        mutag_accuracies.append(final_mutag_acc)

        # record for this fold
        fold_records.append({
            "fold": fold,
            "indices": {"train_idx": list(map(int, train_idx)),
                        "test_idx": list(map(int, test_idx))},
            "history": hist,
            "final_accuracy": float(final_acc),
        })

    # save best model
    torch.save(best_model_state, model_path)

    # save split info for the best model
    with open(split_path, "w") as f:
        json.dump(best_split, f, indent=2)

    # consolidated log with summary + all fold records

    mutag_valid = [a for a in mutag_accuracies if a is not None]
    mutag_mean = float(np.mean(mutag_valid)) if mutag_valid else None
    mutag_std = float(np.std(mutag_valid)) if mutag_valid else None

    summary = {
        "fold_accuracies": [float(a) for a in accuracies],
        "mean_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
        "std_accuracy":  float(np.std(accuracies)) if accuracies else 0.0,

        "mutag_fold_accuracies": [(None if a is None else float(a)) for a in mutag_accuracies],
        "mutag_mean_accuracy": (None if mutag_mean is None else float(mutag_mean)),
        "mutag_std_accuracy": (None if mutag_std is None else float(mutag_std)),

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
            "flip_at": FLIP_AT / 100,
            "stratified": True,
            "split_strategy": "Stratified folds on merged dataset. Extra test eval split.",
            "env": {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda": getattr(torch.version, "cuda", None),
                "cudnn": torch.backends.cudnn.version(),
                "device": str(device)
            },
        },
        "folds": fold_records
    }
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)

    if VERBOSE:
        print(f"\n Average accuracy over {K_FOLDS} folds: {float(np.mean(accuracies)) if accuracies else 0.0: .4f}")
        print(f"Saved best model → {model_path}")
        print(f"Saved log → {log_path}")
