import json

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
import random

from config import *
from model import GAT
from train import train
from evaluate import *
from edit_path_graphs import *

# todo: less test script style, more function building blocks?
#  add logic to evaluate prediction on parameters to be defined (eg. changes per edit distance)
#  add visualizations of decision boundary
#  work on dataset logic. loaded again many times


def train_kcv_mutag():

    """Trains a GAT network with k-fold cross validation on MUTAG. Saves the best performing model over all folds and
    epochs. Logs the best model's training accuracy, training and test split, as well as the test accuracies and
    standard deviation over all folds."""

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    dataset = TUDataset(root=ROOT, name=DATASET_NAME)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    labels = [data.y.item() for data in dataset]  # extract ground truth labels from data set
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)  # define k folds for cross validation
    accuracies = []

    # to track for saving only the best performing model
    best_acc = 0
    best_model_state = None
    best_split = None

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(dataset)), labels)):

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

        # adam optimizer for parameter update
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # train model over epochs
        epoch_test_accuracies = []

        for epoch in range(1, EPOCHS + 1):
            # train and evaluate training step
            loss = train(model, train_loader, optimizer, device)
            acc = evaluate_accuracy(model, test_loader, device)
            epoch_test_accuracies.append(acc)
            if epoch % 10 == 0:
                print(f"epoch {epoch: 03d} | loss: {loss: .4f} | epoch {epoch: 03d} | acc: {acc: .4f}")

            # track best model over folds and epochs
            if acc > best_acc:
                print(f"\n DEBUG: new best is model trained over fold {fold + 1} in epoch {epoch} with acc {acc: .4f}")
                best_acc = acc
                best_model_state = model.state_dict()
                best_fold = fold+1
                best_epoch = epoch
                best_split = {'train_idx': train_idx.tolist(),
                              'test_idx': test_idx.tolist(),
                              'fold': fold + 1,
                              'epoch': epoch}

        # evaluate model trained over full fold
        final_acc = evaluate_accuracy(model, test_loader, device)
        accuracies.append(final_acc)
        print(f"fold {fold + 1} accuracy: {acc: .4f}")

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)

    # save best model
    os.makedirs("model", exist_ok=True)
    torch.save(best_model_state, "model/model.pt")

    # log training, test splits
    with open("model/best_split.json", "wb") as f:
        json.dump(best_split, f, indent=2)

    # log k-cv training statistics
    log = {
        "fold_accuracies": [float(a) for a in accuracies],
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "best_model": best_split
    }
    with open("model/training_log_mutag.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n average accuracy over {K_FOLDS} folds: {np.mean(accuracies): .4f}")


if __name__ == "__main__":

    train_kcv_mutag()

    #dataset = TUDataset(root=ROOT, name=DATASET_NAME)

    #graphs = pyg_to_networkx(dataset)

    #edit_paths_graphs(graphs,
    #                  node_subst_cost,
    #                  edge_subst_cost,
    #                  node_ins_cost,
    #                  edge_ins_cost,
    #                  node_del_cost,
    #                  edge_del_cost)

