import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool


# GAT model using GATv2 layers
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=4, dropout=0.5):
        super().__init__()
        # define GATv2 layers
        self.gat1 = GATv2Conv(in_channels, hidden_channels, heads=heads)
        self.gat2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads)
        self.gat3 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads)
        self.gat4 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads)
        # todo: single output for binary classification? or two classes for 0 / 1 class?
        self.lin = torch.nn.Linear(hidden_channels * heads, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        # gatv2 layers with relu activation
        x = F.relu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.gat2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.gat3(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.gat4(x, edge_index))
        # readout: sum pooling
        x = global_add_pool(x, batch)
        return self.lin(x)


