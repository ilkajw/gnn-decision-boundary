import torch
from torch_geometric.utils import to_networkx, from_networkx
from itertools import combinations
import numpy as np
import networkx as nx
import copy
import os


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

    print("DEBUG: computing ged and edit paths for all graph pairs...")

    # todo: delete cast to list and slice later. only for testing purposes
    for i, j in list(combinations(range(len(graphs)), 2))[:4]:
        g1, g2 = graphs[i], graphs[j]
        try:
            # calculate edit operations between g1, g2
            path, cost = nx.optimal_edit_paths(
                g1, g2,
                node_subst_cost=node_subst_cost,
                node_del_cost=node_del_cost,
                node_ins_cost=node_ins_cost,
                edge_subst_cost=edge_subst_cost,
                edge_del_cost=edge_del_cost,
                edge_ins_cost=edge_ins_cost
            )

            # save costs and edit operations to array
            # todo: needed?
            # all_edit_paths[(i, j)] = {
            #    'cost': cost,
            #    'path': path
            # }
            print(f"DEBUG: computed ged for graphs {i} and {j}: cost={cost}, steps={len(path)}. Applying edit ops...")

            # construct graphs from edit operations and save to file
            apply_edit_operations(g1, g2, path, graphs=f"g{i}_to_g{j}")

        except Exception as e:
            print(f"DEBUG: failed to compute ged between graphs {i} and {j}: {e}")


def apply_edit_operations(g1, g2, path, graphs="graphs"):

    # copy of original g1 for changes
    current_graph = copy.deepcopy(g1)

    # add attributes on changes for later readability/debug
    nx.set_node_attributes(current_graph, 'unchanged', 'status')

    # counter which equals ged
    step_counter = 1

    # array to save all edit path graphs between g1 and g2
    graph_sequence = []

    for match in path:
        # new snapshot of current graph for changes to be made
        g_step = copy.deepcopy(current_graph)

        # nodes considered at this step
        u, v = match

        # substitution
        if u is not None and v is not None:
            g_step.nodes[u]['status'] = f'substituted_to_{v}'
            # substitute label
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
            g_step.nodes[new_node]['status'] = 'inserted'
            if 'label' in g2.nodes[v]:
                g_step.nodes[new_node]['label'] = g2.nodes[v]['label']

        # add ged=step_counter to graph's meta data
        g_step.graph['edit_step'] = step_counter

        graph_sequence.append(g_step)
        current_graph = g_step
        step_counter += 1

    # convert pickle to pyg format
    pyg_sequence = [from_networkx(g) for g in graph_sequence]

    # save all edit path graphs between g1 , g2 in one file
    output_dir = "data/edit_path_graphs"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{graphs}_sequence.pt")
    torch.save(pyg_sequence, file_path)  # todo: potentially save ged cost as well

    return



