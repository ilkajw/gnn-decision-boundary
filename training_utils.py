import torch.nn.functional as func
import random
import numpy as np


from predict_utils import *
from edit_path_graphs_utils import *
from sklearn.model_selection import StratifiedKFold


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    # train per batch
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()  # reset gradients
        out = model(data.x, data.edge_index, data.batch).view(-1)  # forward pass, results in [batch_size]
        loss = func.binary_cross_entropy_with_logits(out, data.y.float().view(-1))  # take error
        loss.backward()  # take gradients per backpropagation
        optimizer.step()  # update params
        total_loss += loss.item()  # accumulate loss over batches
    return total_loss / max(1, len(loader))  # average loss over batches


# evaluate accuracy
def evaluate_accuracy(model, loader, device, thr=0.5):
    model.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data.x, data.edge_index, data.batch).view(-1)
            probs = torch.sigmoid(logits)
            pred = (probs > thr).long()                         # predicted hard labels
            y_hard = (data.y.float().view(-1) > thr).long()     # target hard labels
            correct += (pred == y_hard).sum().item()
            n += y_hard.numel()
    return correct / max(1, n)


def evaluate_loss(model, loader, device):
    """Eval BCE loss on a loader (averaged over batches) for logging only."""
    model.eval()
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data.x, data.edge_index, data.batch).view(-1)
            y = data.y.float().view(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(1, n_batches)


def train_and_choose_model(dataset, output_dir, model_fname, split_fname, log_fname, verbose=True):
    """
    Trains a GAT network with k-fold cross validation on MUTAG.
        Saves the best performing model over all folds and epochs.
        Logs the best model's training accuracy, training and test split, as well as the test accuracies and
        standard deviation over all folds.

        Args:
            :param dataset: Graph dataset to fit model to.
            :param output_dir: Directory to save weights of best performing model, train and test split the best performing model was trained on,
             and k-fold cross validation training statistics to.
            :param model_fname: Name of file to save weights of best performing model to.
            :param split_fname: Name of file to save training and test split to.
            :param log_fname: Name of file to save k-fold cross validation training statistics.
            :param verbose: If True, training progress is printed.
    """

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    labels = [data.y.item() for data in dataset]  # extract ground truth labels from data set
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)  # define k folds for cross validation
    accuracies = []

    # to track for saving the best performing model
    best_acc = -1.0
    best_model_state = None
    best_split = None

    # collect all fold histories/indices
    fold_records = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(dataset)), labels)):

        if verbose:
            print(f"\n--- fold {fold + 1} ---")

        # split dataset in train and test set
        train_dataset = dataset[train_idx.tolist()]
        test_dataset = dataset[test_idx.tolist()]

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        # init model
        model = GAT(
            in_channels=dataset.num_features,
            hidden_channels=HIDDEN_CHANNELS,
            heads=HEADS,
            dropout=DROPOUT
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # collect history for this fold
        hist = {"train_loss": [], "test_loss": [], "test_acc": [], "num_epochs": EPOCHS}

        # train model over epochs
        epoch_test_accuracies = []
        
        for epoch in range(1, EPOCHS + 1):
            # train and evaluate training step
            train_loss = train_epoch(model, train_loader, optimizer, device)
            test_acc = evaluate_accuracy(model, test_loader, device)
            test_loss = evaluate_loss(model, test_loader, device)  # log only

            epoch_test_accuracies.append(test_acc)

            # record curves
            hist["train_loss"].append(float(train_loss))
            hist["test_loss"].append(float(test_loss))
            hist["test_acc"].append(float(test_acc))

            if verbose:
                if epoch % 10 == 0:
                    print(f"Epoch {epoch: 03d} | train loss: {train_loss: .4f} | test loss: {test_loss: .4f} | test acc: {test_acc: .4f}")

            # track best model over folds and epochs
            if test_acc > best_acc:
                if verbose:
                    print(f"New best is model trained over fold {fold + 1} in epoch {epoch} with test acc {test_acc: .4f}")
                best_acc = test_acc
                best_model_state = model.state_dict()
                best_split = {'train_idx': train_idx.tolist(),
                              'test_idx': test_idx.tolist(),
                              'fold': fold + 1,
                              'epoch': epoch}

        # evaluate model trained over full fold
        final_acc = evaluate_accuracy(model, test_loader, device)
        accuracies.append(final_acc)
        if verbose:
            print(f"Fold {fold + 1} | Accuracy: {test_acc: .4f}")

        fold_records.append({
            "fold": fold + 1,
            "indices": {"train_idx": train_idx.tolist(), "test_idx": test_idx.tolist()},
            "history": hist,
            "final_accuracy": float(final_acc),
        })

    # save best model
    os.makedirs(output_dir, exist_ok=True)
    model_path = f"{output_dir}/{model_fname}"
    torch.save(best_model_state, model_path)

    # log training, test splits
    split_path = f"{output_dir}/{split_fname}"
    with open(split_path, "w") as f:
        json.dump(best_split, f, indent=2)

    # log k-cv training statistics
    log = {
        "fold_accuracies": [float(a) for a in accuracies],
        "mean_accuracy": np.mean(accuracies),
        "std_accuracy": np.std(accuracies),
        "best_model": best_split,
        "folds": fold_records
    }
    log_path = f"{output_dir}/{log_fname}"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    if verbose:
        print(f"\n Average accuracy over {K_FOLDS} folds: {np.mean(accuracies): .4f}")
