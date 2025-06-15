import torch


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch).view(-1)
            pred = (out > 0).long()  # todo: suitable criterion for 1 or 0?
            correct += (pred == data.y).sum().item()  # sum up correct predictions
    return correct / len(loader.dataset)  # return accuracy
