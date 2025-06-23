import numpy as np
import os
import pickle
import networkx as nx
import gedlibpy
from itertools import combinations
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import TUDataset


# todo: this does not work yet as gedlibpy not compatible on windows in its org form
def extract_labels(G):
    for u, attrs in G.nodes(data=True):
        attrs['label'] = int(np.argmax(attrs['x']))
    for u, v, attrs in G.edges(data=True):
        attrs['label'] = int(np.argmax(attrs['edge_attr']))


def apply_node_mapping(g1, g2, node_map):
    current_graph = g1.copy()
    nx.set_node_attributes(current_graph, 'unchanged', 'status')
    step_counter = 1
    graph_sequence = []

    for u, v in node_map.items():
        g_step = current_graph.copy()

        if u != -1 and v != -1:
            g_step.nodes[u]['status'] = f'substituted_to_{v}'
            g_step.nodes[u]['label'] = g2.nodes[v]['label']
        elif u != -1 and v == -1:
            if u in g_step:
                g_step.remove_node(u)
        elif u == -1 and v != -1:
            new_node = f'v{v}'
            g_step.add_node(new_node)
            g_step.nodes[new_node]['status'] = 'inserted'
            g_step.nodes[new_node]['label'] = g2.nodes[v]['label']

        g_step.graph['edit_step'] = step_counter
        graph_sequence.append(g_step)
        current_graph = g_step
        step_counter += 1

    return graph_sequence


def run_batch_ged_and_save(dataset, output_dir="gedlib_output"):
    os.makedirs(output_dir, exist_ok=True)
    gedlibpy.PyInitEnv()
    gedlibpy.PySetEditCost("CHEM_1")
    gedlibpy.PySetMethod("IPFP", "")
    gedlibpy.PyInitMethod()

    graphs = []
    graph_ids = []

    for i, data in enumerate(dataset):
        g_nx = to_networkx(data, to_undirected=True, node_attrs=['x'], edge_attrs=['edge_attr'])
        extract_labels(g_nx)
        graph_id = gedlibpy.PyAddNXGraph(g_nx)
        graphs.append(g_nx)
        graph_ids.append(graph_id)

    for i, j in combinations(range(len(graphs)), 2):
        id1, id2 = graph_ids[i], graph_ids[j]
        gedlibpy.PyRunMethod(id1, id2)
        cost = gedlibpy.PyGetUpperBound(id1, id2)
        node_map = gedlibpy.PyGetNodeMap(id1, id2)

        g1, g2 = graphs[i], graphs[j]
        edit_path_graphs = apply_node_mapping(g1, g2, node_map)

        # Save all graphs into a single file using pickle
        out_path = os.path.join(output_dir, f"g{i}_to_g{j}_sequence.pkl")
        with open(out_path, 'wb') as f:
            pickle.dump(edit_path_graphs, f)

        # Also save cost for reference
        cost_path = os.path.join(output_dir, f"g{i}_to_g{j}_cost.txt")
        with open(cost_path, 'w') as f:
            f.write(str(cost))

        print(f"Saved GED={cost:.2f} and {len(edit_path_graphs)} edit graphs for g{i} to g{j}.")
