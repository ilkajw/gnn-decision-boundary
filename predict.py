from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from config import *
from model import GAT
from edit_path_graphs import *
import pickle


def mutag_predictions():
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
    model.load_state_dict(torch.load("model/model.pt"))
    model.eval()

    predictions = {}
    correct_class_0 = {}
    correct_class_1 = {}

    for idx, data in enumerate(loader):
        data = data.to(device)
        out = model(data)
        probs = torch.sigmoid(out.view(-1))
        pred = (probs > 0.5).long()
        true = data.y.item()
        correct = int(pred == true)

        predictions[idx] = {
            'true_label': true,
            'pred_label': pred,
            'correct': correct
        }
        # Separate correctly classified graphs
        if correct:
            if true == 0:
                correct_class_0[idx] = predictions[idx]
            elif true == 1:
                correct_class_1[idx] = predictions[idx]

    os.makedirs("data/predictions", exist_ok=True)

    with open("data/predictions/mutag_predictions.json", "wb") as f:
        pickle.dump(predictions, f)



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
        save_path = os.path.join(save_dir, f"predictions_g{i}_to_g{j}.pt") # todo: pt? not json?
        with open(save_path, "wb") as f:
            pickle.dump(preds, f)


        print(f"DEBUG: saved predictions for path {i} → {j} to {save_path}")
