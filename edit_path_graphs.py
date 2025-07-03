import networkx as nx
from torch_geometric.utils import from_networkx


def generate_all_edit_path_graph_sequences(db_name="MUTAG", seed=42):
    """
    Generates all intermediate NetworkX graphs from stored edit paths.

    Returns:
        A dict of {(i, j, iteration): [graph_0, graph_1, ..., graph_k]} for all pairs and iterations.
    """

    # todo: import repo
    graph_dataset = GraphDataset(root="data", name=db_name, from_existing_data="TUDataset")
    graph_dataset.create_nx_graphs()
    nx_graphs = graph_dataset.nx_graphs

    edit_paths = load_edit_paths_from_file(db_name=db_name, file_path="data")
    if edit_paths is None:
        raise RuntimeError("Edit paths file missing or unreadable.")

    sequence_dict = dict()
    for (i, j), paths in edit_paths.items():
        for ep in paths:
            sequence = ep.create_edit_path_graphs(nx_graphs[i], nx_graphs[j], seed=seed)
            key = (i, j, ep.iteration)
            sequence_dict[key] = sequence
    return sequence_dict


def add_edit_steps_to_sequence_graphs(sequence_dict):
    """
    Adds 'edit_step' metadata to each graph in each sequence within the sequence dictionary.

    Args:
        sequence_dict (dict): {(i, j, iteration): [graph_0, graph_1, ..., graph_k]}

    Returns:
        dict: The same dictionary with 'edit_step' added to each graph's metadata.
    """
    for key, sequence in sequence_dict.items():
        for step, g in enumerate(sequence):
            g.graph['edit_step'] = step  # 0-based
    return sequence_dict


def filter_connected_graphs(sequence_dict):
    """
    Filters each graph sequence to include only connected graphs,
    preserving the original structure and metadata.

    Args:
        sequence_dict (dict): {(i, j, iteration): [graph_0, graph_1, ..., graph_k]}

    Returns:
        dict: Filtered version of sequence_dict with only connected graphs.
    """
    filtered_dict = {}

    for key, sequence in sequence_dict.items():
        connected_sequence = [
            g for g in sequence if nx.is_connected(g)
        ]
        filtered_dict[key] = connected_sequence

    return filtered_dict


def nx_sequences_to_pyg(sequence_dict):
    """
     Converts a dictionary of NetworkX graph sequences to PyTorch Geometric format,
     preserving node/edge attributes and graph-level metadata (e.g., edit_step).

     Args:
         sequence_dict (dict): {(i, j, iteration): [graph_0, graph_1, ..., graph_k]}

     Returns:
         dict: {(i, j, iteration): [pyg_graph_0, pyg_graph_1, ..., pyg_graph_k]}
     """

    pyg_sequence_dict = {}

    for key, sequence in sequence_dict.items():
        pyg_sequence = []
        for g in sequence:
            # Convert to PyG, automatically includes node and edge attributes
            pyg_g = from_networkx(g)

            # Manually add graph-level metadata (edit step)
            for meta_key, meta_val in g.graph.items():
                setattr(pyg_g, meta_key, meta_val)

            pyg_sequence.append(pyg_g)

        pyg_sequence_dict[key] = pyg_sequence

    return pyg_sequence_dict


