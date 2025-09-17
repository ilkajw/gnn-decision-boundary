"""
Generate per-graph predictions with a trained GNN on a PyG TU dataset.

Loads a saved model state, runs inference over the entire dataset, and writes a JSON
mapping from graph index to prediction metadata:
``{"true_label": int, "pred_label": int, "correct": bool}``.

Configuration is read from ``config`` (e.g., ``ROOT``, ``DATASET_NAME``, ``MODEL``,
``MODEL_DIR``, ``PREDICTIONS_DIR``, ``MODEL_CLS``, ``MODEL_KWARGS``).

Notes
-----
- Assumes **binary graph-level classification** with one logit per graph; logits
  are passed through ``sigmoid`` and thresholded at ``0.5``.
- Uses ``batch_size=1`` to keep dataset indices aligned with output keys.
"""
import os
import json
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from config import ROOT, MODEL, MODEL_DIR, DATASET_NAME, PREDICTIONS_DIR, MODEL_CLS, MODEL_KWARGS

# --- Set input, output paths ---
model_path = os.path.join(MODEL_DIR, f"{DATASET_NAME}_{MODEL}_model.pt")

model_cls = MODEL_CLS
model_kwargs = MODEL_KWARGS

output_dir = PREDICTIONS_DIR
output_fname = f"{DATASET_NAME}_{MODEL}_predictions.json"


# --- Function definition ---
def dataset_predictions(model_cls,
                        model_kwargs,
                        output_dir,
                        output_fname,
                        model_path,
                        ):
    """
    Run inference on a PyG TU dataset and save **per-graph** predictions to JSON.

    Initializes ``model_cls(in_channels=dataset.num_features, **model_kwargs)``,
    loads weights from ``model_path``, iterates the dataset with ``batch_size=1``,
    and records true labels, predicted labels (thresholded at 0.5), and correctness.

    :param model_cls: Class or callable that constructs the model; must accept
        ``in_channels`` and ``**model_kwargs`` and return an ``nn.Module``.
    :type model_cls: type[nn.Module] | collections.abc.Callable[..., nn.Module]
    :param model_kwargs: Keyword arguments forwarded to ``model_cls``.
    :type model_kwargs: dict
    :param output_dir: Directory where the predictions JSON will be saved
        (created if it does not exist).
    :type output_dir: str | os.PathLike
    :param output_fname: File name of the predictions JSON.
    :type output_fname: str
    :param model_path: Path to the saved model weights (``state_dict``).
    :type model_path: str | os.PathLike

    :returns: ``None``. Writes predictions to ``output_dir / output_fname``.
    :rtype: None

    **Assumptions**
        - Binary graph-level classification with **one logit per graph**.
        - Threshold fixed at ``0.5`` for converting probabilities to labels.
        - Dataset is loaded as ``TUDataset(root=ROOT, name=DATASET_NAME)``.

    **Output JSON structure**
        ``{
            "<graph_index>": {
                "true_label": int,
                "pred_label": int,
                "correct": bool
            },
            ...
        }``
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = TUDataset(root=ROOT, name=DATASET_NAME)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)  # batch_size=1 to keep index tracking

    # Initialize model and load state_dict from previous training
    model = model_cls(
        in_channels=dataset.num_features,
        **model_kwargs
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    predictions = {}
    #  Record model predictions and true labels for every graph in dataset
    with torch.no_grad():
        for idx, data in enumerate(loader):
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            probs = torch.sigmoid(out.view(-1))
            pred = (probs > 0.5).long()
            true = int(data.y.item())
            correct = int(pred == true)  # TODO: can probs be deleted, not used

            predictions[idx] = {
                'true_label': int(true),
                'pred_label': int(pred.item()),
                'correct': bool(correct)  # TODO: can probs be deleted, not used
            }
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_fname)
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)


# --- Run ---
if __name__ == "__main__":

    dataset_predictions(
        model_cls=model_cls,
        model_kwargs=model_kwargs,
        output_dir=output_dir,
        output_fname=output_fname,
        model_path=model_path
    )
