import sys
import os

# add submodule root to Python path
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
    Adds source node, target node, edit step to graph metadata for later analysis.
    Converts nx graphs to pyg.
    Saves each pyg graph sequence identified by source graph, target graph, iteration to .pt file.

    Args:
        db_name (str): Dataset name.
        seed (int): Random seed for edit path generation.
        data_dir (str): Directory where edit paths are stored.
        output_dir (str): Where to save the PyG graphs.
        fully_connected (bool): If True, filter only connected graphs.
    """

    os.makedirs(output_dir, exist_ok=True)

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

    # for reconstruction of node feature tensor x
    num_node_classes = dataset.unique_node_labels
    print(f"Number of node classes, equivalent to size of original feature tensor: {num_node_classes} \n ")

    # todo: for debugging only. delete afterwards
    counter = 0
    max_label = 0

    # create graphs from operations per (source graph, target graph)
    for (i, j), paths in edit_paths.items():

        # todo: right now only 1 path per pair. if not potentially more for other datasets, delete inner loop
        # loop through all paths between graph pairs i, j
        for ep in paths:

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
                sequence = [g for g in sequence if nx.is_connected(g)]

            # drop edge attrs, convert to pyg objects, copy metadata
            pyg_sequence = []

            for step, g in enumerate(sequence):

                # todo: alternatively copy edge attrs from g and transform back to vectors. else check for
                #  any use of edge attrs later on
                # strip edge attributes as not used for learning
                g_no_edge_attrs = nx.Graph()


                # add nodes with attr tensor x reconstructed from nx graph's scalar primary_label
                for n, d in g.nodes(data=True):
                    label = d['primary_label']
                    if label >= num_node_classes:
                        print(f"Node {n} of edit path graph {g.graph['edit_step']} between graphs {i}, {j} with "
                              f"label {label} >= {num_node_classes}. ")
                        counter += 1

                    if label > max_label:
                        max_label = label

                    # todo: causes error as label >= num_node_classes for many nodes
                    #d['x'] = F.one_hot(torch.tensor(label), num_classes=num_node_classes).float()
                    #g_no_edge_attrs.add_node(n, **d)

                # add edges without attributes
                #g_no_edge_attrs.add_edges_from(g.edges())

                # convert to PyG
                #pyg_g = from_networkx(g_no_edge_attrs)

                # copy metadata
                #for meta_key, meta_val in g.graph.items():
                #    setattr(pyg_g, meta_key, meta_val)
                #pyg_sequence.append(pyg_g)

    print(f"Number of nodes with label >= {num_node_classes}: {counter} \n Max label: {max_label}")
            # save sequence to file
            #file_path = os.path.join(output_dir, f"g{i}_to_g{j}_it{ep.iteration}_graph_sequence.pt")
            #torch.save(pyg_sequence, file_path)


