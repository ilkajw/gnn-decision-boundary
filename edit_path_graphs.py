import sys
import os

# Add submodule root to Python path
submodule_path = os.path.abspath("external")
if submodule_path not in sys.path:
    sys.path.insert(0, submodule_path)

import networkx as nx
import torch
import torch.nn.functional as F
from pg_gnn_edit_paths.utils.io import load_edit_paths_from_file
from pg_gnn_edit_paths.utils.GraphLoader.GraphLoader import GraphDataset
from torch_geometric.utils import from_networkx



def generate_and_save_all_edit_path_graphs(db_name="MUTAG",
                                           seed=42,
                                           data_dir="data",
                                           output_dir="data/pyg_edit_path_graphs",
                                           fully_connected=True):
    """
    Generates all nx graphs from edit path sequences.
    Optionally filters for fully connected graphs only.
    Adds metadata to graph for later filtering.
    Converts nx graphs to pyg.
    Saves each pyg graph sequence identified by source graph, target graph, iteration to one .pt file.

    Args:
        db_name (str): Dataset name.
        seed (int): Random seed for edit path generation.
        data_dir (str): Directory where edit paths are stored.
        output_dir (str): Where to save the PyG graphs.
        fully_connected (bool): If True, filter only connected graphs.
    """

    # load nx graphs from original dataset
    dataset = GraphDataset(root=data_dir,
                           name=db_name,
                           from_existing_data="TUDataset",
                           task="graph_classification")

    dataset.create_nx_graphs()
    nx_graphs = dataset.nx_graphs
    # todo: this doesnt work yet. label of nx graph nodes coming out of repos create_edit_path_graphs
    #  is bigger than unique_node_labels
    num_node_classes = dataset.unique_node_labels  # for reconstruction of node feature tensor

    # load edit paths
    edit_paths = load_edit_paths_from_file(db_name=db_name, file_path=data_dir)
    if edit_paths is None:
        raise RuntimeError("Edit path file not found. Run generation first.")

    os.makedirs(output_dir, exist_ok=True)

    for (i, j), paths in edit_paths.items():
        # loop through all paths between graph pairs i, j
        for ep in paths:
            print(f"DEBUG: going into path {ep.iteration} between {i}, {j} ")

            # create edit path graph sequence for path iteration between i, j
            sequence = ep.create_edit_path_graphs(nx_graphs[i], nx_graphs[j], seed=seed)

            # assign metadata to nx graphs
            for step, g in enumerate(sequence):
                g.graph['edit_step'] = step
                g.graph['source_idx'] = i
                g.graph['target_idx'] = j
                g.graph['iteration'] = ep.iteration
                g.graph['distance'] = ep.distance

            # filter for fully connected graphs
            if fully_connected:
                print(f"DEBUG: filtering for fully connected")
                sequence = [g for g in sequence if nx.is_connected(g)]

            # convert nx objects to pyg objects and copy metadata
            pyg_sequence = []
            for step, g in enumerate(sequence):
                print(f"DEBUG: converting graph {step} of curr sequence")

                # strip edge attributes bc not used for learning and throwing error
                # todo: alternatively copy edge attrs from g. if not check if edge attrs are used later on
                g_no_edge_attrs = nx.Graph()

                # add nodes with attr tensor x reconstructed from nx graphs's scalar primary_label
                # todo: this causes an error as num_node_classes >= label for some nodes
                for n, d in g.nodes(data=True):
                    label = d['primary_label']
                    d['x'] = F.one_hot(torch.tensor(label), num_classes=num_node_classes).float()
                    g_no_edge_attrs.add_node(n, **d)

                # add edges without attributes
                g_no_edge_attrs.add_edges_from(g.edges())

                # convert to PyG
                pyg_g = from_networkx(g_no_edge_attrs)

                # copy metadata
                for meta_key, meta_val in g.graph.items():
                    setattr(pyg_g, meta_key, meta_val)
                print(f"added meta data to graph {step} of curr sequence")
                pyg_sequence.append(pyg_g)

            # save sequence to file
            print(f"DEBUG: will save pyg graphs from sequence {ep.iteration} between {i}, {j}")
            file_path = os.path.join(output_dir, f"g{i}_to_g{j}_it{ep.iteration}_graph_sequence.pt")
            torch.save(pyg_sequence, file_path)


