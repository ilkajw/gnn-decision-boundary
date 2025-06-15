import torch
import numpy as np
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold

from config import *
from model import GAT
from train import train
from evaluate import evaluate


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = TUDataset(root=ROOT, name=DATASET_NAME)
    labels = [data.y.item() for data in dataset]
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    accuracies = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(dataset)), labels)):
        print(f"\n--- Fold {fold + 1} ---")

        train_dataset = dataset[train_idx.tolist()]
        test_dataset = dataset[test_idx.tolist()]
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        model = GAT(
            in_channels=dataset.num_features,
            hidden_channels=HIDDEN_CHANNELS,
            heads=HEADS
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        for epoch in range(1, EPOCHS + 1):
            loss = train(model, train_loader, optimizer, device)
            if epoch % 10 == 0:
                print(f"Epoch {epoch: 03d} | Loss: {loss: .4f}")

        acc = evaluate(model, test_loader, device)
        accuracies.append(acc)
        print(f"Fold {fold + 1} Accuracy: {acc: .4f}")

    print(f"\n Average Accuracy over {K_FOLDS} folds: {np.mean(accuracies): .4f}")


if __name__ == "__main__":
    main()