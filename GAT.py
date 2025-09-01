import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, global_add_pool, global_mean_pool, global_max_pool

# GAT model using GATv2 layers for dynamic attention

# supported activations
_ACTS = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
    "identity": nn.Identity,
}


class GAT(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_layers: int = 4,
            hidden_channels: int = 8,
            heads: int = 8,
            activation: str = "elu",  # 'relu' | 'elu' | 'gelu' | 'leaky_relu' | 'identity'
            dropout: float = 0.2,
            readout: str = "sum",  # 'sum' | 'mean' | 'max'
            mlp_layers: int = 1,
            out_channels: int = 1,  # graph-level logits dim
            ):

        super().__init__()
        assert in_channels >= 1, "in_channels must be >= 1"
        assert hidden_channels >= 1, "hidden_channels must be >= 1"
        assert num_layers >= 1, "num_layers must be >= 1"
        assert out_channels >= 1, "out_channels must be >= 1"
        assert 0 <= dropout <= 1, "dropout must be in [0, 1]"
        assert readout in ("mean", "sum", "max"), f"unknown readout '{readout}'. choose 'mean', 'sum' or 'max'."
        assert mlp_layers >= 1, "mlp_layers must be >= 1"
        assert activation.lower() in _ACTS, f"unknown activation '{activation.lower()}'. choose from {list(_ACTS.keys())}."

        act_cls = _ACTS[activation.lower()]  # class
        self.act = act_cls()  # instance
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # --- GATv2 stack ---
        # todo: adapt to gcn and graphsage stack construction
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads, concat=False))
        else:
            self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads, concat=True))
            for _ in range(num_layers - 2):
                self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, concat=True))
            self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, concat=False))

        # --- graph presentation readout ---
        self.readout = {
            "sum": global_add_pool,
            "mean": global_mean_pool,
            "max": global_max_pool,
        }[readout]

        # --- head ---
        if mlp_layers == 1:
            self.head = nn.Linear(hidden_channels, out_channels)
        else:
            blocks: list[nn.Module] = []
            for _ in range(mlp_layers - 1):
                blocks += [nn.Linear(hidden_channels, hidden_channels),
                           act_cls(),  # instantiate per block for safety
                           nn.Dropout(dropout) if dropout > 0 else nn.Identity()
                           ]
            blocks += [nn.Linear(hidden_channels, out_channels)]
            self.head = nn.Sequential(*blocks)

    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.act(x)  # activation
            # no dropout for last conv
            if i < len(self.convs)-1:
                x = self.drop(x)
        g = self.readout(x, batch)
        return self.head(g)
