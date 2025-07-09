import torch


# evaluate accuracy on mutag
def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            probs = torch.sigmoid(out.view(-1))
            pred = (probs > 0.5).long()
            correct += (pred == data.y).sum().item()  # sum up correct predictions
    return correct / len(loader.dataset)  # return accuracy

