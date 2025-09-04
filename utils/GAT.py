import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, global_add_pool, global_mean_pool, global_max_pool

# GAT model using GATv2 layers for dynamic attention

# Supported activations
_ACTS = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
    "identity": nn.Identity,
}


class GAT(torch.nn.Module):
    """
        Graph Attention Network (GATv2) for graph classification.

        Builds a stack of :class:`torch_geometric.nn.GATv2Conv` layers with optional
        multi-head attention, aggregates node embeddings with a graph readout
        (``'sum'``, ``'mean'``, or ``'max'``), and feeds the result through an MLP head
        to produce per-graph outputs.

        :param in_channels: Input node feature dimension.
        :type in_channels: int
        :param num_layers: Number of GATv2 convolutional layers (≥1). Default: ``4``.
        :type num_layers: int
        :param hidden_channels: Hidden feature size per head. Default: ``8``.
        :type hidden_channels: int
        :param heads: Number of attention heads in each GATv2 layer. Default: ``8``.
        :type heads: int
        :param activation: Nonlinearity after each convolution
            (``'relu'``, ``'elu'``, ``'gelu'``, ``'leaky_relu'``, ``'identity'``).
            Default: ``'elu'``.
        :type activation: str
        :param dropout: Drop probability applied after each conv (except the last),
            and between MLP layers when ``mlp_layers > 1``. Default: ``0.2``.
        :type dropout: float
        :param readout: Graph-level aggregation function (``'sum'``, ``'mean'``,
            or ``'max'``). Default: ``'sum'``.
        :type readout: str
        :param mlp_layers: Number of linear blocks in the output head. If ``1``,
            the head is a single ``Linear``; otherwise it is
            ``[Linear → Activation → Dropout] × (mlp_layers-1)`` followed by a final
            ``Linear``. Default: ``1``.
        :type mlp_layers: int
        :param out_channels: Output dimension per graph (e.g., number of classes or 1 for a logit).
            Default: ``1``.
        :type out_channels: int

        **Inputs**
            - **x** (*torch.Tensor*): Node features of shape ``[N, in_channels]``.
            - **edge_index** (*torch.LongTensor*): COO edge index of shape ``[2, E]``.
            - **batch** (*torch.LongTensor*): Graph id for each node of shape ``[N]``.

        **Returns**
            - *torch.Tensor*: Graph-level predictions of shape ``[num_graphs, out_channels]``.

        **Notes**
            - For ``num_layers > 1``, hidden GATv2 layers use ``concat=True`` so the
              output size is ``hidden_channels * heads``; the **last** layer uses
              ``concat=False`` to return ``hidden_channels`` before readout.
            - GATv2 computes dynamic (order-invariant) attention coefficients compared
              to the original GAT formulation.

        **Example**
            >>> model = GAT(in_channels=16, hidden_channels=32, heads=4,
            ...             num_layers=3, readout='mean', out_channels=2)
            >>> out = model(x, edge_index, batch)   # shape: (num_graphs, 2)
    """

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
            out_channels: int = 1,  # Graph-level logits dim
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
        # TODO: adapt to GCN and GraphSAGE stack construction
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads, concat=False))
        else:
            self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads, concat=True))
            for _ in range(num_layers - 2):
                self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, concat=True))
            self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, concat=False))

        # --- Graph readout ---
        self.readout = {
            "sum": global_add_pool,
            "mean": global_mean_pool,
            "max": global_max_pool,
        }[readout]

        # --- Head ---
        if mlp_layers == 1:
            self.head = nn.Linear(hidden_channels, out_channels)
        else:
            blocks: list[nn.Module] = []
            for _ in range(mlp_layers - 1):
                blocks += [nn.Linear(hidden_channels, hidden_channels),
                           act_cls(),  # Instantiate activation per block for safety
                           nn.Dropout(dropout) if dropout > 0 else nn.Identity()
                           ]
            blocks += [nn.Linear(hidden_channels, out_channels)]
            self.head = nn.Sequential(*blocks)

    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.act(x)  # Activation
            if i < len(self.convs)-1:  # No dropout for last convolution
                x = self.drop(x)
        g = self.readout(x, batch)
        return self.head(g)
