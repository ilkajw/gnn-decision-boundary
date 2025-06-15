import networkx as nx
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from itertools import combinations
from apply_edit_ops import apply_edit_operations
import os
import numpy as np

from config import ROOT, DATASET_NAME

# todo: encapsulate script style into functions. fuse two edit path files

# define cost function for ged
def node_subst_cost(u_attrs, v_attrs):
    return 0 if np.array_equal(u_attrs.get('label'), v_attrs.get('label')) else 1


def edge_subst_cost(e1_attrs, e2_attrs):
    return 0 if np.array_equal(e1_attrs.get('label'), e2_attrs.get('label')) else 1

node_del_cost = lambda attrs: 1
node_ins_cost = lambda attrs: 1
edge_del_cost = lambda attrs: 1
edge_ins_cost = lambda attrs: 1



# load MUTAG dataset
dataset = TUDataset(root=ROOT, name=DATASET_NAME)

# convert each pyg data object from MUTAG to networkx including labels
graphs = []
for data in dataset:
    g_nx = to_networkx(data, to_undirected=True, node_attrs=['x'], edge_attrs=['edge_attr'])

    # todo: try if comparisons with label vectors work well
    # flatten attributes to label per one hot encoding for better comparison in ged computation
    #for node in g_nx.nodes():
    #    x = g_nx.nodes[node].get('x')
    #    if x is not None:
    #        g_nx.nodes[node]['label'] = int(x.argmax().item())  # one-hot to label

    #for u, v, attrs in g_nx.edges(data=True):
    #    ea = attrs.get('edge_attr')
    #    if ea is not None:
    #        g_nx.edges[u, v]['label'] = int(ea.argmax().item())  # one-hot to label

    graphs.append(g_nx)

print(f"DEBUG: converted {len(graphs)}/{len(dataset)} graphs to networkx format.")


# create dir for apply_edit_operations() to save constructed graphs to
output_root = "edit_path_graphs"
os.makedirs(output_root, exist_ok=True)


all_edit_paths = {}
print("DEBUG: computing ged and edit paths for all graph pairs...")

# calculate optimal edit paths and graph edit distance,
# construct edit path graphs from edit operations
for i, j in combinations(range(len(graphs)), 2):
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
        all_edit_paths[(i, j)] = {
            'cost': cost,
            'path': path
        }
        print(f"DEBUG: computed ged for graphs {i} and {j}: cost={cost}, steps={len(path)}")

        # construct graphs from edit operations
        apply_edit_operations(g1, g2, path, graphs=f"g{i}_to_g{j}")

    except Exception as e:
        print(f"DEBUG: failed to compute ged between graphs {i} and {j}: {e}")
