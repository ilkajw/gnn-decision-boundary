import torch
from torch import nn
from torch.nn import LayerNorm
from torch_geometric.nn import GCNConv, global_add_pool, BatchNorm, global_mean_pool, global_max_pool

# supported activations
_ACTS = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
    "identity": nn.Identity,
}

# supported normalisations
_NORMS = {
    None: None,  # no norm
    "batch": BatchNorm,
    "layer": LayerNorm,
}


class GCN(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            num_layers: int = 4,
            activation: str = "elu",  # 'relu' | 'elu' | 'gelu' | 'leaky_relu' | 'identity'
            normalization: str | None = None,
            dropout=0.2,
            readout: str = "sum",  # 'sum' | 'mean' | 'max'
            mlp_layers: int = 1,
            out_channels: int = 1,  # graph-level logits dim
            improved: bool = False,

    ):
        super().__init__()
        assert in_channels >= 1, "in_channels must be >= 1"
        assert hidden_channels >= 1, "hidden_channels must be >= 1"
        assert num_layers >= 1, "num_layers must be >= 1"
        assert activation.lower() in _ACTS, f"unknown activation '{activation.lower()}'. " \
                                            f"choose from {list(_ACTS.keys())}."
        assert normalization is None or normalization.lower() in _NORMS, f"unknown normalization '{normalization.lower()}'. " \
                                                f"choose from {list(_NORMS.keys())}."
        assert 0 <= dropout <= 1, "dropout has to be in range [0, 1]"
        assert readout in ["sum", "mean", "max"], f"unknown readout '{readout}'. choose 'mean', 'sum' or 'max'."
        assert mlp_layers >= 1, "mlp_layers must be >= 1"
        assert out_channels >= 1, "out_channels must be >= 1"

        act_cls = _ACTS[activation.lower()]
        norm_cls = _NORMS[normalization]

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList() if normalization is not None else None
        self.act = act_cls()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.readout = {
            "sum": global_add_pool,
            "mean": global_mean_pool,
            "max": global_max_pool,
        }[readout]

        # define per-layer input and output dims
        in_dims = [in_channels] + [hidden_channels] * max(0, num_layers - 1)
        out_dims = [hidden_channels] * max(0, num_layers - 1) + [hidden_channels] if num_layers > 1 else [
            hidden_channels]

        # GCN stack
        for in_d, out_d in zip(in_dims, out_dims):
            self.convs.append(GCNConv(in_d, out_d, improved=improved))
            if self.norms is not None:
                self.norms.append(norm_cls(out_d))

        # final head
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

    def forward(self, x, edge_index, batch, edge_weight=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if self.norms is not None:
                x = self.norms[i](x)
            x = self.act(x)
            # no dropout after last conv
            if i < len(self.convs) - 1:
                x = self.drop(x)
        g = self.readout(x, batch)
        return self.head(g)
