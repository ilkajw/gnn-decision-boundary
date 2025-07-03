import os

import networkx as nx
import torch
from torch_geometric.utils import from_networkx




def generate_and_save_all_graphs(db_name="MUTAG",
                                 seed=42,
                                 data_dir="data",
                                 output_dir="data/pyg_edit_path_graphs",
                                 fully_connected=True):
    """
    Generates all nx graphs from edit path sequences.
    Optionally filters for fully connected graphs only.
    Adds metadata for later filtering.
    Converts to pyg objects.
    Saves each pyg graph sequence identified by source graph, target graph, iteration to one .pt file.

    Args:
        db_name (str): Dataset name.
        seed (int): Random seed for edit path generation.
        data_dir (str): Directory where edit paths are stored.
        output_dir (str): Where to save the PyG graphs.
        fully_connected (bool): If True, filter only connected graphs.
    """

    # load nx graphs from original dataset
    dataset = GraphDataset(root=data_dir, name=db_name, from_existing_data="TUDataset")
    dataset.create_nx_graphs()
    nx_graphs = dataset.nx_graphs

    # load edit paths
    edit_paths = load_edit_paths_from_file(db_name=db_name, file_path=data_dir)
    if edit_paths is None:
        raise RuntimeError("Edit path file not found. Run generation first.")

    os.makedirs(output_dir, exist_ok=True)

    for (i, j), paths in edit_paths.items():

        # loop through all paths between i, j
        for ep in paths:

            # create edit path graph sequence for path 'ep' between i, j
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

            # convert to pyg with metadata
            pyg_sequence = []

            # convert nx objects to pyg and copy metadata
            for step, g in enumerate(sequence):
                pyg_g = from_networkx(g)
                for meta_key, meta_val in g.graph.items():
                    setattr(pyg_g, meta_key, meta_val)
                pyg_sequence.append(pyg_g)

            # save sequence to file
            file_path = os.path.join(output_dir, f"g{i}_to_g{j}_it{ep.iteration}_graph_sequence.pt")
            torch.save(pyg_sequence, file_path)


