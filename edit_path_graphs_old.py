import torch
from torch_geometric.utils import to_networkx, from_networkx
from itertools import combinations
import numpy as np
import networkx as nx
import copy
import os
from utils.io import load_edit_paths_from_file
from utils.GraphLoader.GraphLoader import GraphDataset


# define cost function for ged
def node_subst_cost(u_attrs, v_attrs):
    v1 = np.array(u_attrs.get('label'))
    v2 = np.array(v_attrs.get('label'))
    return np.linalg.norm(v1 - v2)


def edge_subst_cost(e1_attrs, e2_attrs):
    v1 = np.array(e1_attrs.get('label'))
    v2 = np.array(e2_attrs.get('label'))
    return np.linalg.norm(v1 - v2)


node_del_cost = lambda attrs: 1
node_ins_cost = lambda attrs: 1
edge_del_cost = lambda attrs: 1
edge_ins_cost = lambda attrs: 1


# convert each pyg data object from MUTAG to networkx with node, edge labels
def pyg_to_networkx(dataset):

    graphs = []
    for data in dataset:
        g_nx = to_networkx(data, to_undirected=True, node_attrs=['x'], edge_attrs=['edge_attr'])

        for node in g_nx.nodes():
            x = g_nx.nodes[node].get('x')
            if x is not None:
                g_nx.nodes[node]['label'] = tuple(x)  # label for ged function

        for u, v, attrs in g_nx.edges(data=True):
            edge_attr = attrs.get('edge_attr')
            if edge_attr is not None:
                g_nx.edges[u, v]['label'] = tuple(edge_attr)  # label for ged function

        graphs.append(g_nx)
        print(f"DEBUG: converted {len(graphs)}/{len(dataset)} graphs to networkx format.")

    return graphs


def edit_paths_graphs(graphs,
                    node_subst_cost,
                    edge_subst_cost,
                    node_ins_cost,
                    edge_ins_cost,
                    node_del_cost,
                    edge_del_cost):

    """ Calculates the graph edit distance between all pairwise combinations in 'graphs'. Constructs the edit path
    graphs from the path. Saves all graphs from one path as pyg objects to .pt file."""

    print("DEBUG: computing ged and edit paths for all graph pairs...")

    # todo: delete cast to list and slice later. only for testing purposes
    for i, j in list(combinations(range(len(graphs)), 2))[:4]:
        g1, g2 = graphs[i], graphs[j]
        try:
            # calculate edit operations between g1, g2
            ops_path, cost = nx.optimal_edit_paths(
                g1, g2,
                node_subst_cost=node_subst_cost,
                node_del_cost=node_del_cost,
                node_ins_cost=node_ins_cost,
                edge_subst_cost=edge_subst_cost,
                edge_del_cost=edge_del_cost,
                edge_ins_cost=edge_ins_cost
            )

            # save costs and edit operations to array
            # all_edit_paths[(i, j)] = {
            #    'cost': cost,
            #    'path': path
            # }
            print(f"DEBUG: computed ged for graphs {i} and {j}: cost={cost}, steps={len(ops_path)}. Applying edit ops...")

            # construct graphs from edit operations and save to file
            construct_graphs_from_path(g1, g2, ops_path, pair=f"g{i}_to_g{j}")

        except Exception as e:
            print(f"DEBUG: failed to compute ged between graphs {i} and {j}: {e}")



def construct_graphs_from_path(g1, g2, path, pair="graphs", connected_only=False):

    """Constructs networkx graphs from path between g1, g2.
    Converts them to pyg format and saves to a .pt file."""

    # copy of original g1 for changes
    current_graph = copy.deepcopy(g1)

    # counter which equals ged
    step_counter = 1

    # array to save all edit path graphs between g1 and g2
    graph_sequence = []

    for match in path:
        g_step = copy.deepcopy(current_graph)  # snapshot of current graph for changes to be made
        u, v = match

        # substitution
        if u is not None and v is not None:
            if 'label' in g2.nodes[v]:
                g_step.nodes[u]['label'] = g2.nodes[v]['label']

        # deletion
        elif u is not None and v is None:
            if u in g_step:
                g_step.remove_node(u)

        # insertion
        elif u is None and v is not None:
            new_node = max(g_step.nodes) + 1 if len(g_step.nodes) > 0 else 0  # todo: check if this works well as integer, else string?
            g_step.add_node(new_node)
            if 'label' in g2.nodes[v]:
                g_step.nodes[new_node]['label'] = g2.nodes[v]['label']

        # add ged=step_counter to graph's meta data
        g_step.graph['edit_step'] = step_counter

        graph_sequence.append(g_step)
        current_graph = g_step
        step_counter += 1

    if connected_only:
        graph_sequence = [graph for graph in graph_sequence if nx.is_connected(graph)]

    # convert networkx to pyg format
    pyg_sequence = [from_networkx(g) for g in graph_sequence]  # todo: check if edit_step=step_counter is converted too

    # save all edit path graphs between g1 , g2 in one file
    output_dir = "data/edit_path_graphs"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{pair}_sequence.pt")
    torch.save(pyg_sequence, file_path)

    return




