import torch


# evaluate accuracy on mutag
def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch).view(-1)
            pred = (out > 0).long()  # todo: suitable criterion for 1 or 0?
            correct += (pred == data.y).sum().item()  # sum up correct predictions
    return correct / len(loader.dataset)  # return accuracy


# todo: probably delete as not needed anymore
def predict(model, loader, device):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            preds = out.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(data.y.cpu())
    return torch.cat(all_preds), torch.cat(all_labels)
