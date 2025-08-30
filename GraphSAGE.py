# graphsage_graph.py
# GraphSAGE for graph-level classification.

from __future__ import annotations
from typing import Optional

import torch
from torch import nn, Tensor
from torch_geometric.nn import SAGEConv, global_mean_pool, global_add_pool, global_max_pool


class MLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        assert num_layers >= 1
        act = activation if activation is not None else nn.ReLU()

        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(in_channels, out_channels))
        else:
            layers.append(nn.Linear(in_channels, hidden_channels))
            layers.append(act)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_channels, hidden_channels))
                layers.append(act)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_channels, out_channels))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class GraphSAGE(nn.Module):
    """
    GraphSAGE (Hamilton et al., 2017) using torch_geometric.nn.SAGEConv.

    Graph-level only:
      x, edge_index, batch -> graph logits [num_graphs, out_channels]

    Args:
        in_channels:       Input node feature dim.
        hidden_channels:   Hidden dim for SAGE layers.
        out_channels:      Output dim (#classes or 1).
        num_layers:        Number of SAGE layers (>=1).
        dropout:           Dropout after each SAGE layer except the last.
        residual:          Residual connections when dims match.
        norm:              Normalization layer class (e.g., nn.BatchNorm1d or nn.LayerNorm) or None.
        jk:                'last' or 'concat' (jumping knowledge across layers).
        readout:           'mean' | 'sum' | 'max' (graph pooling).
        mlp_layers:        Layers in final MLP head (>=1). If 1, it’s a single Linear.
        activation:        Activation module (default ReLU).
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.2,
        residual: bool = True,
        norm: Optional[type[nn.Module]] = nn.BatchNorm1d,
        jk: str = "last",                 # or "concat"
        readout: str = "mean",            # 'mean' | 'sum' | 'max'
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"
        assert jk in ("last", "concat")
        assert readout in ("mean", "sum", "max")

        self.dropout_p = dropout
        self.residual = residual
        self.jk = jk
        self.act = activation if activation is not None else nn.ReLU()

        # Convs + optional norms
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList() if norm is not None else None

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        if self.norms is not None:
            self.norms.append(norm(hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            if self.norms is not None:
                self.norms.append(norm(hidden_channels))

        # representation dim after JK
        rep_dim = hidden_channels if jk == "last" else hidden_channels * num_layers

        # final classifier, 1 or several layers
        self.head = nn.Linear(rep_dim, out_channels)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # pool
        if readout == "mean":
            self.pool = global_mean_pool
        elif readout == "sum":
            self.pool = global_add_pool
        else:
            self.pool = global_max_pool

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.norms is not None:
            for n in self.norms:
                if hasattr(n, "reset_parameters"):
                    n.reset_parameters()
        if hasattr(self.head, "reset_parameters"):
            self.head.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        """
        Returns graph-level logits: [num_graphs, out_channels]
        """
        layer_outs = []
        h = x
        for i, conv in enumerate(self.convs):
            h_in = h
            h = conv(h, edge_index)
            if self.norms is not None:
                h = self.norms[i](h)
            h = self.act(h)
            if i < len(self.convs) - 1:
                h = self.dropout(h)
            if self.residual and h.shape == h_in.shape:
                h = h + h_in
            layer_outs.append(h)

        rep = layer_outs[-1] if self.jk == "last" else torch.cat(layer_outs, dim=-1)
        g = self.pool(rep, batch)
        return self.head(g)

    @torch.no_grad()
    def predict_proba_graph(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        """
        Probabilities from logits (sigmoid for binary, softmax for multiclass).
        """
        logits = self.forward(x, edge_index, batch)
        if logits.size(-1) == 1:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=-1)

    def compute_loss(self, logits: Tensor, target: Tensor) -> Tensor:
        """
        Convenience loss:
          - Binary: BCEWithLogitsLoss
          - Multiclass: CrossEntropyLoss
        """
        if logits.size(-1) == 1:
            y = target.float().view_as(logits)
            return nn.BCEWithLogitsLoss()(logits, y)
        return nn.CrossEntropyLoss()(logits, target.view(-1))
