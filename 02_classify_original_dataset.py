import os
import json
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from config import *
from model import GAT

# --- config ---

output_dir = f"data_control/{DATASET_NAME}/predictions/"
output_fname = f"{DATASET_NAME}_predictions.json"
model_path = "model_control/model.pt"

# --- function definition ---

def dataset_predictions(output_dir,
                        output_fname,
                        model_path="model_control/model.pt",
                        ):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load dataset
    dataset = TUDataset(root=ROOT, name=DATASET_NAME)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)  # batch_size=1 to keep index tracking

    # re-instantiate model
    model = GAT(
        in_channels=dataset.num_features,
        hidden_channels=HIDDEN_CHANNELS,
        heads=HEADS,
        dropout=DROPOUT
    ).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    predictions = {}

    for idx, data in enumerate(loader):
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        probs = torch.sigmoid(out.view(-1))
        pred = (probs > 0.5).long()
        true = data.y.item()
        correct = int(pred == true)

        predictions[idx] = {
            'true_label': int(true),
            'pred_label': int(pred.item()),
            'correct': bool(correct)
        }

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_fname)
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)


# --- run ---
if __name__ == "__main__":

    dataset_predictions(output_dir=output_dir,
                        output_fname=output_fname,
                        model_path=model_path
    )
