import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # for cuBLAS determinism
os.environ["PYTHONHASHSEED"] = "42"
import json
import numpy as np

import torch
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)

from time import perf_counter
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold

from EditPathGraphDataset import FlatGraphDataset
from GAT import GAT
from training_utils import train_epoch, evaluate_accuracy, set_seed, evaluate_loss
from config import (
    DATASET_NAME, EPOCHS, LEARNING_RATE, BATCH_SIZE,
    HIDDEN_CHANNELS, HEADS, DROPOUT, K_FOLDS, LABEL_MODE
)


# ----------------------- helpers -----------------------

def infer_in_channels(dataset):
    sample = dataset[0]
    return sample.x.size(-1)


def detect_origin_indices(dataset):
    """
    Return (original_idx, extra_idx) using data.origin.
    Raises error if any datapoint is missing .origin.
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
            raise RuntimeError(f"Data object at index {i} has invalid origin='{g.origin}' (expected 'org' or 'edit')")
    return original_idx, extra_idx


def hard_labels(dataset, indices=None):
    """
    Hard 0/1 labels for StratifiedKFold; robust to soft targets.
    If indices is provided, only collect labels for those indices.
    """
    if indices is None:
        indices = range(len(dataset))
    ys = []
    for i in indices:
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
    set_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # merged dataset input path
    merged_pt = f"data/{DATASET_NAME}/processed/{DATASET_NAME}_merged_dataset_tagged.pt"

    # outputs
    out_dir = f"model_merged_test_org_only/{DATASET_NAME}"
    os.makedirs(out_dir, exist_ok=True)
    model_fname = f"{DATASET_NAME}_model_merged_test_org_only.pt"
    split_fname = "train_test_split_merged_test_org_only.json"
    log_fname = "train_log_merged_test_org_only.json"
    model_path = os.path.join(out_dir, model_fname)
    split_path = os.path.join(out_dir, split_fname)
    log_path = os.path.join(out_dir, log_fname)

    # load merged dataset (original + edits, collated)
    dataset = FlatGraphDataset(saved_path=merged_pt, verbose=True)

    # detect original vs edit indices via .origin field
    original_idx, edit_idx = detect_origin_indices(dataset)
    assert len(original_idx) > 0, "No 'org' samples found. Check tagging in the merged dataset."
    assert len(edit_idx) > 0, "No 'edit' samples found. Check tagging in the merged dataset."
    if VERBOSE:
        print(f"Detected {len(original_idx)} original graphs and {len(edit_idx)} edit graphs.")

    # Stratified K-fold only on original subset
    labels_original = hard_labels(dataset, original_idx)
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_channels = infer_in_channels(dataset)

    # track best model by test accuracy across all folds/epochs
    accuracies = []
    best_acc = -1.0
    best_model_state = None
    best_split = None

    # collect all folds' info
    fold_records = []
    orig_array = np.array(original_idx)

    for fold, (train_pos, test_pos) in enumerate(skf.split(np.zeros(len(orig_array)), labels_original), start=1):
        if VERBOSE:
            print(f"\n--- fold {fold} ---")

        train_orig_idx = orig_array[train_pos].tolist()
        test_orig_idx = orig_array[test_pos].tolist()

        # train = original_train_fold + all edits
        train_indices_all = train_orig_idx + edit_idx

        train_subset = Subset(dataset, train_indices_all)
        test_subset = Subset(dataset, test_orig_idx)

        g = torch.Generator()
        g.manual_seed(42)
        train_loader = DataLoader(train_subset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  generator=g,
                                  num_workers=0,
                                  persistent_workers=False
                                  )
        test_loader = DataLoader(test_subset,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False,
                                 num_workers=0,
                                 persistent_workers=False
                                 )

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
                print(f"Epoch {epoch:03d} | train loss: {train_loss:.4f} | "
                      f"test loss: {test_loss:.4f} | test acc: {test_acc:.4f} | "
                      f"time: {int(m):02d}:{s:06.3f}")

            if round(float(test_acc), 6) > round(float(best_acc), 6):
                if VERBOSE:
                    print(f"New best: fold {fold}, epoch {epoch}, acc={test_acc:.4f}")
                best_acc = float(test_acc)
                best_model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                best_split = {
                    'train_idx': [int(i) for i in train_indices_all],
                    'test_idx':  [int(i) for i in test_orig_idx],
                    'fold': fold,
                    'epoch': epoch,
                    'note': 'train = org train fold + all edits; test = org test fold'
                }

        final_acc = evaluate_accuracy(model, test_loader, device)
        accuracies.append(final_acc)
        if VERBOSE:
            print(f"Fold {fold} | Final Accuracy: {final_acc:.4f}")

        fold_records.append({
            "fold": fold,
            "indices": {
                "train_idx": [int(i) for i in train_indices_all],
                "test_idx":  [int(i) for i in test_orig_idx]
            },
            "history": hist,
            "final_accuracy": float(final_acc),
        })

    torch.save(best_model_state, model_path)
    with open(split_path, "w") as f:
        json.dump(best_split, f, indent=2)

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
            "label_mode": LABEL_MODE,
            "split_strategy": "Stratified folds on original subset only; edits added to train",
            "env": {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda": getattr(torch.version, "cuda", None),
                "cudnn": torch.backends.cudnn.version(),
                "device": str(device)
             },
        },
        "sizes": {"original": len(original_idx), "extra": len(edit_idx)},
        "folds": fold_records
    }
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)

    if VERBOSE:
        print(f"\nAverage accuracy over {K_FOLDS} folds: {float(np.mean(accuracies)):.4f}")
        print(f"Saved best model → {model_path}")
        print(f"Saved log → {log_path}")
