import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":16:8"  # for reproducibility
import torch
import json
import random
import numpy as np
import torch.nn.functional as func
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from config import K_FOLDS, MODEL, DATASET_NAME, BATCH_SIZE, LEARNING_RATE, EPOCHS, MODEL_CLS, MODEL_KWARGS


def setup_reproducibility(seed=42):
    """
    Configure deterministic behavior across Python, NumPy, and PyTorch.

    Sets seeds and toggles PyTorch/cuDNN flags to maximize run-to-run
    reproducibility on CPU and GPU.

    :param seed: Seed used for Python, NumPy, and PyTorch RNGs. Default: ``42``.
    :type seed: int
    """

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, device):
    """
    Performs one train loop over ``loader`` with ``binary_cross_entropy_with_logits``
    loss on the model's graph-level logits.

    :param model: PyTorch module taking ``(x, edge_index, batch)`` and returning
        graph-level logits. Output is reshaped with ``.view(-1)``.
    :type model: torch.nn.Module
    :param loader: DataLoader yielding batches with attributes
        ``x``, ``edge_index``, ``batch``, and ``y``.
    :type loader: torch.utils.data.DataLoader
    :param optimizer: Optimizer whose ``zero_grad()``, ``step()`` are called.
    :type optimizer: torch.optim.Optimizer
    :param device: Device to move batches to (e.g., ``'cuda'`` or ``'cpu'``).
    :type device: torch.device | str

    :returns: Mean loss over all batches in the epoch.
    :rtype: float
    """
    model.train()
    total_loss = 0
    # Loop over batches in loader
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()  # Reset gradients
        # todo: make compatible to logit dim > 1
        logits = model(data.x, data.edge_index, data.batch).view(-1)  # Forward pass, flattened to [batch_size]
        y_true = data.y.float().view(-1)  # Flattened classes
        loss = func.binary_cross_entropy_with_logits(logits, y_true)
        loss.backward()  # Take gradients per backprop
        optimizer.step()  # Update weights
        total_loss += loss.item()  # accumulate loss over batches
    return total_loss / max(1, len(loader))  # Average loss over batches


def evaluate_accuracy(model, loader, device, thr=0.5):
    """
    Evaluate accuracy for binary graph-level classification.

    Runs the model in eval mode with ``torch.no_grad()``, converts logits to
    probabilities via ``sigmoid``, thresholds at ``thr`` to get hard predictions,
    and compares them to hard targets.

    :param model: PyTorch module taking ``(x, edge_index, batch)`` and returning
                  graph-level logits. Output is reshaped with ``.view(-1)``.
    :type model: torch.nn.Module
    :param loader: DataLoader yielding batches with attributes
                   ``x``, ``edge_index``, ``batch``, and ``y``.
    :type loader: torch.utils.data.DataLoader
    :param device: Device to move batches to (e.g., ``'cuda'`` or ``'cpu'``).
    :type device: torch.device | str
    :param thr: Probability threshold for positive class (``0 < thr < 1``).
                Default: ``0.5``.
    :type thr: float

    :returns: Accuracy in ``[0, 1]`` over all graphs in ``loader``.
    :rtype: float
    """
    model.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            # TODO: make compatible to logit dim > 1
            logits = model(data.x, data.edge_index, data.batch).view(-1)  # Forward pass, flattened to [batch_size]
            probs = torch.sigmoid(logits)
            pred = (probs > thr).long()   # Hard labels by thresholding predictions
            y_true = data.y.float().view(-1)
            y_hard = (y_true > thr).long()  # Hard labels for soft labeling in augmented data
            correct += (pred == y_hard).sum().item()
            n += y_hard.numel()
    return correct / max(1, n)  # Average accuracy over datapoints

# TODO: delete? just for logging
def evaluate_loss(model, loader, device):
    """Eval BCE loss on a loader (averaged over batches) for logging only."""
    model.eval()
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            # todo: make compatible to logit dim > 1
            logits = model(data.x, data.edge_index, data.batch).view(-1)  # forward pass, flattened to [batch_size]
            y_true = data.y.float().view(-1)
            loss = func.binary_cross_entropy_with_logits(logits, y_true)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(1, n_batches)


def train_model_kcv(
        model_cls,  # GAT | GCN | GraphSAGE class from config
        model_kwargs,
        dataset,
        output_dir,
        model_fname,
        split_fname,
        log_fname,
        verbose=True
):
    """
    Train a GNN with k-fold cross-validation on a PyG dataset.

    Runs training/evaluation across folds, selects the best-performing model
    (over all folds and epochs) by test accuracy, and writes artifacts to disk:
    model weights, best train/test split, cross-validation summary.

    :param dataset: PyG dataset to cross-validate on.
    :type dataset: torch_geometric.data.InMemoryDataset | torch_geometric.data.Dataset
    :param output_dir: Directory where artifacts are saved (created if missing).
    :type output_dir: str | os.PathLike
    :param model_fname: File name for the best model weights (``state_dict``).
    :type model_fname: str
    :param split_fname: File name for the JSON containing the best train/test split.
    :type split_fname: str
    :param log_fname: File name for the JSON with per-fold metrics and best fold/epoch.
    :type log_fname: str
    :param model_cls: Class or callable that constructs the model, called as
                      ``model_cls(in_channels=dataset.num_features, **model_kwargs)``.
    :type model_cls: type[nn.Module] | collections.abc.Callable[..., nn.Module]
    :param model_kwargs: Keyword arguments forwarded to ``model_cls``.
    :type model_kwargs: dict
    :param verbose: If ``True``, print progress messages.
    :type verbose: bool

    :returns: ``None``. Artifacts are written to ``output_dir``.
    :rtype: None
    """

    setup_reproducibility()
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, model_fname)
    split_path = os.path.join(output_dir, split_fname)
    log_path = os.path.join(output_dir, log_fname)

    model_kwargs = model_kwargs or {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    true_labels = [data.y.item() for data in dataset]
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    accuracies = []

    # Init to track best model
    best_acc = -1.0
    best_model_state = None
    best_split = None

    # To collect per-fold histories
    fold_records = []

    print(f"--- original {MODEL} training on {DATASET_NAME} ---")

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(dataset)), true_labels)):

        if verbose:
            print(f"\n--- fold {fold + 1} ---")

        train_dataset = dataset[train_idx.tolist()]
        test_dataset = dataset[test_idx.tolist()]

        # For reproducibility
        g = torch.Generator()
        g.manual_seed(42)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=g,
            num_workers=0,
            persistent_workers=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            persistent_workers=False
        )

        # Instantiate fresh model for each fold
        model = model_cls(
            in_channels=dataset.num_features,
            **model_kwargs
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # To collect per-fold history
        # todo: give more detailed history from training utils
        hist = {"train_loss": [], "test_loss": [], "test_acc": [], "num_epochs": EPOCHS}

        # Train over epochs
        epoch_test_accuracies = []
        for epoch in range(1, EPOCHS + 1):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            test_acc = evaluate_accuracy(model, test_loader, device)
            test_loss = evaluate_loss(model, test_loader, device)

            epoch_test_accuracies.append(test_acc)
            hist["train_loss"].append(float(train_loss))
            hist["test_loss"].append(float(test_loss))
            hist["test_acc"].append(float(test_acc))

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch: 03d} | train loss: {train_loss: .4f} | "
                      f"test loss: {test_loss: .4f} | test acc: {test_acc: .4f}")

            # Track best across all folds and epochs
            if round(float(test_acc), 6) > round(float(best_acc), 6):
                if verbose:
                    print(f"New best is model trained over fold {fold + 1} "
                          f"in epoch {epoch} with test acc {test_acc: .4f}")
                best_acc = float(test_acc)
                # Clone for deep copy
                best_model_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
                torch.save(best_model_state, model_path)

                best_split = {
                    'train_idx': train_idx.tolist(),
                    'test_idx': test_idx.tolist(),
                    'fold': fold + 1,
                    'epoch': epoch,
                }

        # Evaluate final model of this fold
        final_acc = evaluate_accuracy(model, test_loader, device)
        accuracies.append(final_acc)
        if verbose:
            print(f"Fold {fold + 1} | Accuracy: {final_acc: .4f}")

        fold_records.append({
            "fold": fold + 1,
            "indices": {"train_idx": train_idx.tolist(), "test_idx": test_idx.tolist()},
            "history": hist,
            "final_accuracy": float(final_acc),
        })

    # ---- Save ----
    os.makedirs(output_dir, exist_ok=True)

    torch.save(best_model_state, model_path)

    with open(split_path, "w") as f:
        json.dump(best_split, f, indent=2)

    # Summary
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
            "augmentation": "None. Original training",
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

    if verbose:
        print(f"\n Average test accuracy over {K_FOLDS} folds: {np.mean(accuracies): .4f}")
