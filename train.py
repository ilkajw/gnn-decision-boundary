import torch
import torch.nn.functional as F

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()  # reset gradients
        out = model(data.x, data.edge_index, data.batch).view(-1)  # forward pass, results in [batch_size]
        loss = F.binary_cross_entropy_with_logits(out, data.y.float())  # take error
        loss.backward()  # take gradients per backpropagation
        optimizer.step()  # update params
        total_loss += loss.item()  # accumulate loss in epoch
    return total_loss / len(loader)  # average loss
