"""
Generate NetworkX and PyG graph sequences from precomputed edit-path operations.

Loads edit-paths produced by the external utilities, reconstructs intermediate
NetworkX graphs for each (source, target, iteration), optionally filters for
connected graphs, drops edge attributes, converts to PyTorch-Geometric objects,
and writes outputs:

 - NX sequences:  ROOT/DATASET_NAME/nx_edit_path_graphs/*.pkl
 - PyG sequences: ROOT/DATASET_NAME/pyg_edit_path_graphs/*.pt
 - Audit JSONs:   ROOT/DATASET_NAME/test/*.json

Requires: config.py (ROOT, DATASET_NAME, FULLY_CONNECTED_ONLY) and the
external pg_gnn_edit_paths utilities.
"""

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
edit_path_ops_dir = os.path.join("external", "pg_gnn_edit_paths", f"example_paths_{DATASET_NAME}")
nx_output_dir = os.path.join(ROOT, DATASET_NAME, 'nx_edit_path_graphs')
pyg_output_dir = os.path.join(ROOT, DATASET_NAME, 'pyg_edit_path_graphs')
test_output_dir = os.path.join(ROOT, DATASET_NAME, "test")

# --- Function definition ---

def generate_edit_path_graphs(
        db_name,
        data_dir,
        nx_output_dir,
        pyg_output_dir,
        test_output_dir,
        fully_connected_only,
        seed=42,
):
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

    # Load nx graphs from original dataset
    dataset = GraphDataset(root=data_dir,
                           name=db_name,
                           from_existing_data="TUDataset",
                           task="graph_classification")
    dataset.create_nx_graphs()
    nx_graphs = dataset.nx_graphs

    # Load all pre-calculated edit path operations from file
    edit_paths = load_edit_paths_from_file(db_name=db_name, file_path=data_dir)
    if edit_paths is None:
        raise RuntimeError("Edit path file not found. Run generation first.")

    num_node_classes = dataset.unique_node_labels  # for reconstruction of node feature tensor x from integer values

    # To track behaviour
    last_graph_insertions = []
    no_intermediates_before_filter = []
    no_intermediates_after_filter = []

    # Create graphs from operations per (source graph, target graph) = (i, j)
    for (i, j), paths in edit_paths.items():

        for ep in paths:

            # Create edit path graph sequence for path between i, j
            nx_sequence = ep.create_edit_path_graphs(nx_graphs[i], nx_graphs[j], seed=seed)

            if len(nx_sequence) <= 2:
                no_intermediates_before_filter.append((i, j))

            # Check if the target graph is included in sequence
            def node_match(n1, n2):
                return n1['primary_label'] == n2['primary_label']

            def edge_match(e1, e2):
                return e1['label'] == e2['label']

            # TODO: Include edge label comparison in isomorphism test
            #  as soon as edge labels from external repo are consistent

            last_graph = nx_sequence[-1]
            last_and_target_graph_isomorphic = is_isomorphic(last_graph, nx_graphs[j], node_match=node_match)
            if not last_and_target_graph_isomorphic or len(nx_sequence) < 2:
                nx_sequence.append(nx_graphs[j].copy())
                nx_sequence[-1].graph["operation"] = "target_graph_insertion"
                last_graph_insertions.append((i, j))

            num_operations = len(nx_sequence) - 1

            # Assign metadata to each nx graph in sequence
            for step, g in enumerate(nx_sequence):

                # Handle sequences with operation null for their start graphs
                if g.graph["edit_step"] == 0 and "operation" not in g.graph:
                    g.graph["operation"] = "start"

                g.graph['source_idx'] = i
                g.graph['target_idx'] = j
                g.graph['iteration'] = ep.iteration
                g.graph['distance'] = ep.distance
                g.graph['edit_step'] = step
                g.graph['num_operations_incl_insertion'] = num_operations

            # Filter for connected graphs
            if fully_connected_only:
                nx_sequence = [g for g in nx_sequence if nx.is_connected(g)]
                if len(nx_sequence) <= 2:
                    no_intermediates_after_filter.append((i, j))

            # Create equal nx graphs without edge attributes at all due to external repo behaviour with inconsistent
            # edge labeling, convert nx to pyg objects, copy nx attributes to pyg instance
            pyg_sequence = []
            for step, g in enumerate(nx_sequence):

                # Instantiate empty nx graph, add nodes with labels and edges
                g_no_edge_attrs = nx.Graph()

                # During node addition, reconstruct 'x' tensor from scalar 'primary_label'
                for n, d in g.nodes(data=True):
                    label = d['primary_label']
                    d['x'] = func.one_hot(torch.tensor(label), num_classes=num_node_classes).float()
                    g_no_edge_attrs.add_node(n, **d)

                # Add edges without attributes
                g_no_edge_attrs.add_edges_from(g.edges())

                # Convert nx to pyg
                pyg_g = from_networkx(g_no_edge_attrs)

                # Copy attributes to pyg instances
                for attr_key, attr_val in g.graph.items():
                    setattr(pyg_g, attr_key, attr_val)
                pyg_sequence.append(pyg_g)

            # Save nx graph sequence
            nx_out_path = os.path.join(nx_output_dir, f"g{i}_to_g{j}_it{ep.iteration}_graph_sequence.pkl")
            with open(nx_out_path, "wb") as f:
                pickle.dump(nx_sequence, f)

            # Save pyg graph sequence
            pyg_out_path = os.path.join(pyg_output_dir, f"g{i}_to_g{j}_it{ep.iteration}_graph_sequence.pt")
            torch.save(pyg_sequence, pyg_out_path)

    # Save paths with target graph insertion
    os.makedirs(test_output_dir, exist_ok=True)
    with open(os.path.join(test_output_dir, f"{DATASET_NAME}_paths_with_target_graph_inserted.json"), "w") as f:
        json.dump(last_graph_insertions, f, indent=2)

    # Save paths with no intermediate path graphs
    with open(os.path.join(test_output_dir,
                           f"{DATASET_NAME}_no_interm_graphs_at_graph_creation_before_connect_filter.json"), "w") as f:
        json.dump(no_intermediates_before_filter, f, indent=2)

    with open(os.path.join(test_output_dir,
                           f"{DATASET_NAME}_no_interm_graphs_at_graph_creation_after_connect_filter.json"), "w") as f:
        json.dump(no_intermediates_before_filter, f, indent=2)


# --- Run ---
if __name__ == "__main__":

    # Fail fast if input missing
    if not os.path.exists(edit_path_ops_dir):
        raise FileNotFoundError(f"Missing input directory: {edit_path_ops_dir}")

    os.makedirs(nx_output_dir, exist_ok=True)
    os.makedirs(pyg_output_dir, exist_ok=True)

    generate_edit_path_graphs(
        db_name=DATASET_NAME,
        data_dir=edit_path_ops_dir,
        nx_output_dir=nx_output_dir,
        pyg_output_dir=pyg_output_dir,
        test_output_dir=test_output_dir,
        fully_connected_only=FULLY_CONNECTED_ONLY,
        seed=42
    )


