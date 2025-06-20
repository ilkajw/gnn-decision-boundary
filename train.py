import torch.nn.functional as func


def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    # train per batch
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()  # reset gradients
        out = model(data.x, data.edge_index, data.batch).view(-1)  # forward pass, results in [batch_size]
        loss = func.binary_cross_entropy_with_logits(out, data.y.float())  # take error
        loss.backward()  # take gradients per backpropagation
        optimizer.step()  # update params
        total_loss += loss.item()  # accumulate loss over batches
    return total_loss / len(loader)  # average loss over batches
