import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool


# GAT model
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=1):
        super().__init__()
        self.gat1 = GATv2Conv(in_channels, hidden_channels, heads=heads)
        self.gat2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads)
        self.lin = torch.nn.Linear(hidden_channels * heads, 1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.gat2(x, edge_index))
        x = global_add_pool(x, batch)  # readout: sum pooling
        return self.lin(x)

