from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from config import *
from model import GAT
from edit_path_graphs_old import *
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


def edit_path_predictions_from_dict(pyg_sequence_dict, model_path, dataset):
    """
    Runs predictions on a dictionary of PyG graph sequences and logs results.

    Args:
        pyg_sequence_dict (dict): {(i, j, iteration): [pyg_graph_0, ..., pyg_graph_k]}
        model_path (str): Path to the saved GAT model.
        dataset (TUDataset): The dataset to determine feature dimension.

    Returns:
        list: A list of prediction dictionaries with metadata.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = GAT(
        in_channels=dataset.num_features,
        hidden_channels=HIDDEN_CHANNELS,
        heads=HEADS,
        dropout=DROPOUT
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    predictions = []

    for (i, j, iteration), graph_sequence in pyg_sequence_dict.items():
        for graph in graph_sequence:
            graph = graph.to(device)

            with torch.no_grad():
                out = model(graph.x, graph.edge_index, torch.zeros(graph.num_nodes, dtype=torch.long).to(device))
                prob = torch.sigmoid(out.view(-1)).item()
                pred = int(prob > 0.5)

            predictions.append({
                "prediction": pred,
                "probability": prob,
                "edit_step": int(getattr(graph, 'edit_step', -1)),
                "source_idx": i,
                "target_idx": j,
                "iteration": iteration
            })

    return predictions


def postprocess_edit_path_preds():
    """Adds metadata to the edit path predictions for later analysis."""
