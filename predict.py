from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from config import *
from model import GAT
from edit_path_graphs_exact import *
import pickle


def edit_path_predictions():

    input_dir = "data/edit_path_graphs"
    save_dir = "data/edit_path_preds"
    model_path = "model/model.pt"
    os.makedirs(save_dir, exist_ok=True)

    dataset = TUDataset(root=ROOT, name=DATASET_NAME)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # re-instantiate model
    model = GAT(
        in_channels=dataset.num_features,
        hidden_channels=HIDDEN_CHANNELS,
        heads=HEADS,
        dropout=DROPOUT
    ).to(device)
    model.load_state_dict(torch.load(model_path), map_location=device)
    model.eval()

    for i, j in combinations(range(len(dataset)), 2):

        file_path = os.path.join(input_dir, f"g{i}_to_g{j}_sequence.pt")
        if not os.path.exists(file_path):
            print(f"warning: missing file {file_path}, skipping.")
            continue

        # load edit paths graphs from file
        graphs = torch.load(file_path)
        loader = DataLoader(graphs, batch_size=1, shuffle=False)  # batch_size=1 to keep index tracking

        preds = []

        with torch.no_grad():
            for data in loader:
                # predict
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch)
                # todo: potentially adjust if two out nodes
                probs = torch.sigmoid(out.view(-1))
                pred = (probs > 0.5).long()
                preds.append(pred.cpu().item())

        # save results per path
        save_path = os.path.join(save_dir, f"predictions_g{i}_to_g{j}.pt")
        with open(save_path, "wb") as f:
            pickle.dump(preds, f)
        print(f"DEBUG: saved predictions for path {i} → {j} to {save_path}")


def mutag_predictions():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    dataset = TUDataset(root=ROOT, name=DATASET_NAME)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)  # batch_size=1 to keep index tracking

    # re-instantiate model
    model = GAT(
        in_channels=dataset.num_features,
        hidden_channels=HIDDEN_CHANNELS,
        heads=HEADS,
        dropout=DROPOUT
    ).to(device)
    model.load_state_dict(torch.load("model/model.pt"))
    model.eval()

    predictions = {}

    for idx, data in enumerate(loader):
        data = data.to(device)
        out = model(data)
        # todo: potentially adjust if two out nodes
        probs = torch.sigmoid(out.view(-1))
        pred = (probs > 0.5).long()
        true = data.y.item()
        correct = int(pred == true)

        predictions[idx] = {
            'true_label': true,
            'pred_label': pred,
            'correct': correct
        }

    with open("data/mutag_predictions.pkl", "wb") as f:
        pickle.dump(predictions, f)
