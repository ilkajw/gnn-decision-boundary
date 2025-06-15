import torch

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch).view(-1)
            pred = (out > 0).long()
            correct += (pred == data.y).sum().item()
    return correct / len(loader.dataset)