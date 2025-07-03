from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from config import *
from model import GAT
from edit_path_graphs_old import *
import pickle


# todo: change the workings of this to work more similarly to ep predictions?
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
        # separate correctly classified graphs
        if correct:
            if true == 0:
                correct_class_0[idx] = predictions[idx]
            elif true == 1:
                correct_class_1[idx] = predictions[idx]

    os.makedirs("data/predictions", exist_ok=True)

    with open("data/predictions/mutag_predictions.json", "wb") as f:
        pickle.dump(predictions, f)


def edit_path_predictions(model_path, input_dir, output_dir, dataset_name):
    """
    Loads all pyg graph sequences, each indexed by source, target graph and iteration.
    Runs predictions on graphs per sequence.
    Saves graph sequence with per-graph prediction as graph metadata to file.

    Args:
        pyg_sequence_dict (dict): {(i, j, iteration): [pyg_graph_0, ..., pyg_graph_k]}
        model_path (str): Path to the saved GAT model.
        dataset (TUDataset): The dataset to determine feature dimension.

    Returns:
        list: A list of prediction dictionaries with metadata.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TUDataset(root="data", name=dataset_name)
    model = GAT(
        in_channels=dataset.num_features,
        hidden_channels=HIDDEN_CHANNELS,
        heads=HEADS,
        dropout=DROPOUT
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    predictions = []

    for filename in os.listdir(input_dir):
        if not filename.endswith(".pt"):
            continue

        # load ep graph sequence
        path = os.path.join(input_dir, filename)
        graph_sequence = torch.load(path)

        updated_sequence = []

        # predictions for each graph in sequence
        for graph in graph_sequence:
            graph = graph.to(device)
            with torch.no_grad():
                out = model(graph.x, graph.edge_index, torch.zeros(graph.num_nodes, dtype=torch.long).to(device))
                prob = torch.sigmoid(out.view(-1)).item()
                pred = int(prob > 0.5)

            # add prediction and probability to graph metadata
            graph.prediction = pred
            graph.probability = prob

            updated_sequence.append(graph)

            # dict with all preds
            predictions.append({
                "file": filename,
                "source_idx": int(getattr(graph, 'source_idx', -1)),
                "target_idx": int(getattr(graph, 'target_idx', -1)),
                "iteration": int(getattr(graph, 'iteration', -1)),
                "edit_step": int(getattr(graph, 'edit_step', -1)),
                "prediction": pred,
                "probability": prob
            })

        torch.save(updated_sequence, os.path.join(output_dir, filename))

    return predictions

