import torch


def to_float_y(domain_flag: int | None = None):
    """
    Returns a transform that casts y -> float tensor [0,1].
    Optionally adds `data.source = domain_flag` (0 for original dataset, 1 for edit-path).
    """
    def _tf(data):
        # y may be tensor([0]) long or 0-dim, make it float shape [1]
        y = data.y
        if not torch.is_floating_point(y):
            y = y.float()
        if y.dim() == 0:
            y = y.unsqueeze(0)
        data.y = y
        if domain_flag is not None:
            data.source = torch.tensor([domain_flag], dtype=torch.long)
        return data
    return _tf


def drop_edge_attr():
    def _tf(data):
        if 'edge_attr' in data:
            del data.edge_attr  # remove attribute from the Data object
        return data
    return _tf