import os, json, random, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from EditPathGraphDataset import FlatGraphDataset
from training_utils import train_epoch
from model import GAT
from config import DATASET_NAME, EPOCHS, LEARNING_RATE, BATCH_SIZE, HIDDEN_CHANNELS, HEADS, DROPOUT, K_FOLDS


def set_seed(seed=42):
    random.seed(seed),
    np.random.seed(seed),
    torch.manual_seed(seed),
    torch.cuda.manual_seed_all(seed)


def infer_in_channels(dataset):
    sample = dataset[0]
    return sample.x.size(-1)


def evaluate_soft(model, loader, device):
    """Validation pass compatible with soft or hard labels."""
    model.eval()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data.x, data.edge_index, data.batch).view(-1)
            y = data.y.float().view(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y)

            # "hard" accuracy: compare to thresholded targets (even if soft)
            preds = (torch.sigmoid(logits) > 0.5).float()
            hard_targets = (y > 0.5).float()
            acc = (preds == hard_targets).float().mean().item()

            total_loss += loss.item()
            total_acc += acc
            n_batches += 1

    return total_loss / max(1, n_batches), total_acc / max(1, n_batches)


def get_hard_labels_for_stratification(dataset):
    """ Returns a list of hard labels (0/1) for StratifiedKFold. If labels are soft, threshold is 0.5."""
    ys = []
    for i in range(len(dataset)):
        y = dataset[i].y
        # ensure tensor on CPU
        if isinstance(y, torch.Tensor):
            y_val = float(y.view(-1)[0])
        else:
            y_val = float(y)
        ys.append(int(y_val > 0.5))
    return ys


if __name__ == "__main__":

    set_seed(42)

    merged_pt = f"data/{DATASET_NAME}/processed/{DATASET_NAME}_merged_dataset.pt"
    out_dir = f"model_merged/{DATASET_NAME}"
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f"{DATASET_NAME}_model.pt")
    split_path = os.path.join(out_dir, "split.json")
    log_path = os.path.join(out_dir, "train_log.json")

    # load merged dataset (org + edit-path)
    ds = FlatGraphDataset(saved_path=merged_pt, verbose=True)

    # try StratifiedKFold, else KFold
    try:
        y_hard = get_hard_labels_for_stratification(ds)  # hard labels for fold distribution
        skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
        folds = list(skf.split(np.zeros(len(ds)), y_hard))
        stratified = True
    except Exception as e:
        print(f"[WARN] StratifiedKFold failed ({e}). Falling back to plain KFold.")
        kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
        folds = list(kf.split(np.zeros(len(ds))))
        stratified = False

    in_channels = infer_in_channels(ds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_logs = []
    best_global = {
        "val_loss": float("inf"),
        "state_dict": None,
        "fold": None,
        "epoch": None,
    }

    t0 = time.time()
    for fold_idx, (train_idx, test_idx) in enumerate(folds, start=1):

        print(f"\n=== Fold {fold_idx}/{K_FOLDS} ({'Stratified' if stratified else 'KFold'}) ===")

        train_ds = Subset(ds, train_idx)
        test_ds = Subset(ds, test_idx)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        # fresh model per fold
        model = GAT(
            in_channels=in_channels,
            hidden_channels=HIDDEN_CHANNELS,
            heads=HEADS,
            dropout=DROPOUT
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # per-fold tracking
        best_val = {"loss": float("inf"), "epoch": 0}
        hist = {"epoch": [], "train_loss": [], "test_loss": [], "test_acc_hard": []}

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            test_loss, test_acc = evaluate_soft(model, test_loader, device)

            hist["epoch"].append(epoch)
            hist["train_loss"].append(train_loss)
            hist["test_loss"].append(test_loss)
            hist["test_acc_hard"].append(test_acc)

            if test_loss < best_val["loss"]:
                best_val.update({"loss": test_loss, "epoch": epoch})

                # track global best across folds/epochs
                if test_loss < best_global["val_loss"]:
                    best_global.update({
                        "val_loss": test_loss,
                        "state_dict": model.state_dict(),
                        "fold": fold_idx,
                        "epoch": epoch
                    })

            #if epoch % 10 == 0 or epoch == 1 or epoch == EPOCHS:
            print(f"Fold {fold_idx} | Epoch {epoch:03d} | "
                  f"train {train_loss:.4f} | test {test_loss:.4f} | test_acc {test_acc:.3f}")

        # save per-fold info
        fold_dir = os.path.join(out_dir, f"fold_{fold_idx:02d}")
        os.makedirs(fold_dir, exist_ok=True)
        with open(os.path.join(fold_dir, "log.json"), "w") as f:
            json.dump(hist, f, indent=2)
        with open(os.path.join(fold_dir, "indices.json"), "w") as f:
            json.dump({"train_idx": list(map(int, train_idx)),
                       "test_idx": list(map(int, test_idx))}, f, indent=2)

        fold_logs.append({
            "fold": fold_idx,
            "best_test_loss": best_val["loss"],
            "best_epoch": best_val["epoch"],
            "train_size": len(train_idx),
            "test_size": len(test_idx),
        })

    elapsed = time.time() - t0
    print(f"\nAll folds done in {elapsed / 60:.1f} min")
    print(f"Best global model: fold {best_global['fold']} epoch {best_global['epoch']} "
          f"| loss {best_global['val_loss']:.4f}")

    # save global best model + summary
    if best_global["state_dict"] is not None:
        torch.save(best_global["state_dict"], os.path.join(out_dir, "best_model.pt"))

    summary = {
        "fold_results": fold_logs,
        "best_global": {
            "fold": best_global["fold"],
            "epoch": best_global["epoch"],
            "val_loss": best_global["val_loss"],
        },
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
            "stratified": stratified,
        }
    }
    with open(os.path.join(out_dir, "kfold_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved best model → {os.path.join(out_dir, 'best_model.pt')}")
    print(f"Saved per‑fold logs/indices under {out_dir}")
