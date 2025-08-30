import json
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from config import *

# --- config ---
output_dir = PREDICTIONS_DIR
output_fname = f"{DATASET_NAME}_{MODEL}_predictions.json"
model_path = f"{MODEL_DIR}/{MODEL}_model.pt"
model_cls = MODEL_CLS
model_kwargs = MODEL_KWARGS


# --- function definition ---
def dataset_predictions(model_cls,
                        model_kwargs,
                        output_dir,
                        output_fname,
                        model_path,
                        ):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load dataset
    dataset = TUDataset(root=ROOT, name=DATASET_NAME)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)  # batch_size=1 to keep index tracking

    # init model
    model = model_cls(
        in_channels=dataset.num_features,
        **model_kwargs
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    predictions = {}
    with torch.no_grad():
        for idx, data in enumerate(loader):
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            probs = torch.sigmoid(out.view(-1))
            pred = (probs > 0.5).long()
            true = int(data.y.item())
            correct = int(pred == true)  # todo: can probs be deleted, not used

            predictions[idx] = {
                'true_label': int(true),
                'pred_label': int(pred.item()),
                'correct': bool(correct)  # todo: can probs be deleted, not used
            }

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_fname)
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)


# --- run ---
if __name__ == "__main__":

    dataset_predictions(
        model_cls=model_cls,
        model_kwargs=model_kwargs,
        output_dir=output_dir,
        output_fname=output_fname,
        model_path=model_path
    )
