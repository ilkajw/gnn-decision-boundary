import os
import json
import pickle
import torch
import networkx as nx
import torch.nn.functional as func
from torch_geometric.utils import from_networkx
from networkx import is_isomorphic

from config import ROOT, DATASET_NAME, FULLY_CONNECTED_ONLY
from external.pg_gnn_edit_paths.utils.io import load_edit_paths_from_file
from external.pg_gnn_edit_paths.utils.GraphLoader.GraphLoader import GraphDataset


# --- Set input, output paths ---

edit_path_ops_dir = f"external/pg_gnn_edit_paths/example_paths_{DATASET_NAME}"
nx_output_dir = f"{ROOT}/{DATASET_NAME}/nx_edit_path_graphs"
pyg_output_dir = f"{ROOT}/{DATASET_NAME}/pyg_edit_path_graphs"


# --- Function definition ---

def generate_edit_path_graphs(data_dir,
                              nx_output_dir,
                              pyg_output_dir,
                              db_name=DATASET_NAME,
                              seed=42,
                              fully_connected_only=FULLY_CONNECTED_ONLY):
    """
    Generates all NetworkX graphs from edit path sequences.
    Optionally, filters for fully connected graphs only.
    Adds source node, target node, edit step information to graph attributes.
    Converts NetworkX graphs to PyG.
    Saves each NetworkX graph sequence identified by source graph, target graph, iteration to a .pkl file.
    Saves each pyg graph sequence identified by source graph, target graph, iteration to .pt file.

    :param db_name: Dataset name (used to locate edit paths / name outputs).
    :type db_name: str
    :param seed: Random seed used during edit-path generation.
    :type seed: int
    :param data_dir: Directory where the edit-path inputs are stored.
    :type data_dir: str | os.PathLike
    :param pyg_output_dir: Directory where the ``.pt`` files with PyG graphs are written.
    :type pyg_output_dir: str | os.PathLike
    :param nx_output_dir: Directory where the ``.pkl`` files with NetworkX graphs are written.
    :type nx_output_dir: str | os.PathLike
    :param fully_connected_only: If ``True``, keep only graphs that are connected.
    :type fully_connected_only: bool

    :returns: ``None``.
    :rtype: None
    """

    os.makedirs(pyg_output_dir, exist_ok=True)
    os.makedirs(nx_output_dir, exist_ok=True)

    # load nx graphs from original dataset
    dataset = GraphDataset(root=data_dir,
                           name=db_name,
                           from_existing_data="TUDataset",
                           task="graph_classification")
    dataset.create_nx_graphs()
    nx_graphs = dataset.nx_graphs

    # load all pre-calculated edit path operations
    edit_paths = load_edit_paths_from_file(db_name=db_name, file_path=data_dir)
    if edit_paths is None:
        raise RuntimeError("Edit path file not found. Run generation first.")

    num_node_classes = dataset.unique_node_labels  # for reconstruction of node feature tensor x from integer values

    # to track
    last_graph_insertions = []
    no_intermediates = []

    # create graphs from operations per (source graph, target graph) = (i, j)
    for (i, j), paths in edit_paths.items():

        for ep in paths:

            # create edit path graph sequence for path between i, j
            nx_sequence = ep.create_edit_path_graphs(nx_graphs[i], nx_graphs[j], seed=seed)

            def node_match(n1, n2):
                return n1['primary_label'] == n2['primary_label']

            def edge_match(e1, e2):
                return e1['label'] == e2['label']

            # TODO: include edges in isomorphism test
            # check if the target graph is included in sequence
            last_graph = nx_sequence[-1]
            last_and_target_graph_isomorphic = is_isomorphic(last_graph, nx_graphs[j], node_match=node_match)
            if not last_and_target_graph_isomorphic or len(nx_sequence) < 2:
                nx_sequence.append(nx_graphs[j])
                last_graph_insertions.append((i, j))  # track target graph insertions

            # assign metadata to each nx graph in sequence
            for step, g in enumerate(nx_sequence):
                g.graph['edit_step'] = step
                g.graph['source_idx'] = i
                g.graph['target_idx'] = j
                g.graph['iteration'] = ep.iteration
                g.graph['distance'] = ep.distance
                # todo: how handled best?
                if last_and_target_graph_isomorphic:
                    g.graph['num_all_ops'] = len(ep.all_operations)
                else:
                    g.graph['num_all_ops'] = len(ep.all_operations) + 1

            # Filter for connected graphs
            if fully_connected_only:
                nx_sequence = [g for g in nx_sequence if nx.is_connected(g)]
                # todo: distinguish between after connectedness filter and before
                if len(nx_sequence) <= 2:
                    no_intermediates.append((i, j))

            # Drop edge attributes, convert to pyg objects, copy nx attributes to pyg instance
            pyg_sequence = []
            for step, g in enumerate(nx_sequence):

                # Strip edge attributes from all graphs as some are missing (are not used for inference)
                g_no_edge_attrs = nx.Graph()

                # Add nodes with their attr tensor x reconstructed from scalar 'primary_label'
                for n, d in g.nodes(data=True):
                    label = d['primary_label']
                    d['x'] = func.one_hot(torch.tensor(label), num_classes=num_node_classes).float()
                    g_no_edge_attrs.add_node(n, **d)

                # Add edges without attributes
                g_no_edge_attrs.add_edges_from(g.edges())

                # Convert nx to pyg
                pyg_g = from_networkx(g_no_edge_attrs)

                # Copy attrs to pyg instances
                for attr_key, attr_val in g.graph.items():
                    setattr(pyg_g, attr_key, attr_val)
                pyg_sequence.append(pyg_g)

            # Save nx graph sequence
            nx_out_path = os.path.join(nx_output_dir, f"g{i}_to_g{j}_it{ep.iteration}_graph_sequence.pkl")
            with open(nx_out_path, "wb") as f:
                pickle.dump(nx_sequence, f)

            # Save pyg graph sequence
            file_path = os.path.join(pyg_output_dir, f"g{i}_to_g{j}_it{ep.iteration}_graph_sequence.pt")
            torch.save(pyg_sequence, file_path)

    # Save paths with target graph insertion
    os.makedirs(f"{ROOT}/{DATASET_NAME}/test/", exist_ok=True)
    with open(f"{ROOT}/{DATASET_NAME}/test/{DATASET_NAME}_paths_with_target_graph_inserted.json", "w") as f:
        json.dump(last_graph_insertions, f, indent=2)

    # Save paths with no intermediate path graphs
    with open(f"{ROOT}/{DATASET_NAME}/test/"
              f"{DATASET_NAME}_no_intermediate_graphs_at_graph_seq_creation.json", "w") as f:
        json.dump(no_intermediates, f, indent=2)


# --- Run ---
if __name__ == "__main__":

    # Fail fast if input missing
    if not os.path.exists(edit_path_ops_dir):
        raise FileNotFoundError(f"Missing input directory: {edit_path_ops_dir}")

    os.makedirs(nx_output_dir, exist_ok=True)
    os.makedirs(pyg_output_dir, exist_ok=True)

    generate_edit_path_graphs(
        data_dir=edit_path_ops_dir,
        nx_output_dir=nx_output_dir,
        pyg_output_dir=pyg_output_dir,
        db_name=DATASET_NAME,
        fully_connected_only=FULLY_CONNECTED_ONLY,
        seed=42
    )


