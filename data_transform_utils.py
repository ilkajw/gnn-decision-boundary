import torch


def to_float_y():
    """
    Returns a transform that casts y -> float tensor [0,1].
    """
    def _tf(data):
        y = data.y
        if not torch.is_floating_point(y):
            y = y.float()
        if y.dim() == 0:  # make sure it's at least shape [1]
            y = y.unsqueeze(0)
        data.y = y
        return data
    return _tf


def drop_edge_attr():
    def _tf(data):
        if 'edge_attr' in data:
            del data.edge_attr  # remove attribute from the Data object
        return data
    return _tf


def tag_origin(tag: str):
    assert tag in ("org", "edit")
    def _apply(data):
        data.origin = tag           # "org" or "edit"
        data.is_original = 1 if tag == "org" else 0
        return data
    return _apply