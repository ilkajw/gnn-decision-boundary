import torch
import numpy as np
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
import random

from config import *
from model import GAT
from train import train
from evaluate import evaluate

# todo: test script style into functions transform.
#  transform back to pyg format.
#  add save and load model
#  test model on path graphs
#  add visualizations


def org_train_test():

    # seeds for determinism in shuffles and initializations
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = TUDataset(root=ROOT, name=DATASET_NAME)
    labels = [data.y.item() for data in dataset]  # extract ground truth labels from data set
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)  # define k folds for cross validation
    accuracies = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(dataset)), labels)):
        print(f"\n--- fold {fold + 1} ---")
        # split dataset in train and test set
        train_dataset = dataset[train_idx.tolist()]
        test_dataset = dataset[test_idx.tolist()]
        # define loader for mini batches
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        # init model
        model = GAT(
            in_channels=dataset.num_features,
            hidden_channels=HIDDEN_CHANNELS,
            heads=HEADS,
            dropout=DROPOUT
        ).to(device)

        # define adam optimizer for parameter update
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # train model over epochs
        for epoch in range(1, EPOCHS + 1):
            loss = train(model, train_loader, optimizer, device)
            if epoch % 10 == 0:
                print(f"epoch {epoch: 03d} | loss: {loss: .4f}")

        # evaluate trained model
        acc = evaluate(model, test_loader, device)
        accuracies.append(acc)
        print(f"fold {fold + 1} accuracy: {acc: .4f}")

    print(f"\n average accuracy over {K_FOLDS} folds: {np.mean(accuracies): .4f}")


if __name__ == "__main__":
    org_train_test()
