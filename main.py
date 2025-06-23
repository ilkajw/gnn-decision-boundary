from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
import random

from config import *
from model import GAT
from train import train
from evaluate import *
from edit_path_graphs_exact import *

# todo: less test script style, more function building blocks?
#  add logic to evaluate prediction on parameters to be defined (eg. changes per edit distance)
#  add visualizations of decision boundary
#  work on dataset logic. loaded again many times


def train_test_mutag():

    # seeds for reproducibility in shuffles and initializations
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = TUDataset(root=ROOT, name=DATASET_NAME)
    labels = [data.y.item() for data in dataset]  # extract ground truth labels from data set
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)  # define k folds for cross validation
    accuracies = []

    # todo: only use if all models should be saved
    # directory for saving models trained on mutag
    # save_dir = "models"
    # os.makedirs(save_dir, exist_ok=True)

    best_acc = 0

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(dataset)), labels)):

        print(f"\n--- fold {fold + 1} ---")

        # split dataset in train and test set
        train_dataset = dataset[train_idx.tolist()]
        test_dataset = dataset[test_idx.tolist()]

        # define loaders for mini batches
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
        for epoch in range(1, EPOCHS + 1):
            loss = train(model, train_loader, optimizer, device)
            if epoch % 10 == 0:
                print(f"epoch {epoch: 03d} | loss: {loss: .4f}")

        # evaluate trained model
        acc = evaluate_accuracy(model, test_loader, device)
        accuracies.append(acc)
        print(f"fold {fold + 1} accuracy: {acc: .4f}")

        # track best model
        if acc > best_acc:
            print(f"\n DEBUG: new best is model trained over fold {fold+1}")
            best_acc = acc
            best_model = model

    # todo: potentially change to saving all models earlier
    # save best model
    os.makedirs("model", exist_ok=True)
    torch.save(best_model.state_dict(), "model/model.pt")

    print(f"\n average accuracy over {K_FOLDS} folds: {np.mean(accuracies): .4f}")


if __name__ == "__main__":

    train_test_mutag()

    #dataset = TUDataset(root=ROOT, name=DATASET_NAME)

    #graphs = pyg_to_networkx(dataset)

    #edit_paths_graphs(graphs,
    #                  node_subst_cost,
    #                  edge_subst_cost,
    #                  node_ins_cost,
    #                  edge_ins_cost,
    #                  node_del_cost,
    #                  edge_del_cost)

