# add submodule root to Python path
import os
import sys

submodule_path = os.path.abspath("../external")
if submodule_path not in sys.path:
    sys.path.insert(0, submodule_path)

import json

from pg_gnn_edit_paths.utils.io import load_edit_paths_from_file
from networkx import is_isomorphic
from utils.GraphLoader.GraphLoader import GraphDataset


if __name__ == "__main__":

    # load edit path operations
    edit_paths = load_edit_paths_from_file(db_name='MUTAG',
                                           file_path="../external/pg_gnn_edit_paths/example_paths_MUTAG")

    # load nx graphs from original dataset
    dataset = GraphDataset(root="external/pg_gnn_edit_paths/example_paths_MUTAG",
                           name='MUTAG',
                           from_existing_data="TUDataset",
                           task="graph_classification")

    dataset.create_nx_graphs()
    nx_graphs = dataset.nx_graphs

    # init lists to collect info
    last_graph_missing = []
    last_graph_incl = []
    diff_missing = []
    diff_included = []

    # loop through all pairs of graphs and paths between them (only one right now)
    for (i, j), paths in edit_paths.items():

        for ep in paths:

            # create edit path graph sequence for path iteration between i, j
            sequence = ep.create_edit_path_graphs(nx_graphs[i], nx_graphs[j], seed=42)

            # check isomorphism of last graph and target graph
            last_graph = sequence[-1]
            
            def node_match(n1, n2):
                return n1['primary_label'] == n2['primary_label']

            last_graph_included = is_isomorphic(last_graph, nx_graphs[j], node_match=node_match)

            # collect info on sequences without target graph,
            # and len graph sequence vs num operations (should be diff 1)
            if not last_graph_included:
                last_graph_missing.append(((i, j), len(ep.all_operations), len(sequence)))
                if len(ep.all_operations) + 1 != len(sequence):
                    diff_missing.append(((i, j), len(ep.all_operations), len(sequence)))

            else:
                last_graph_incl.append(((i, j), len(ep.all_operations), len(sequence)))
                if len(ep.all_operations) + 1 != len(sequence):
                    diff_included.append(((i, j), len(ep.all_operations), len(sequence)))


    print(f"Last graph missing: {len(last_graph_missing)} times")
    print(f"Last graph included: {len(last_graph_incl)} times")

    print(f"len_graph_seq != num_ops + 1 while target graph missing: {len(diff_missing)} times")
    print(f"len_graph_seq != num_ops + 1 while target graph included: {len(diff_included)} times")

    # get dicts from lists
    last_graph_missing_dict = {
        f"{i},{j}": {
            "len_all_operations": len_ops,
            "len_graph_sequence": len_seq
        }
        for (i, j), len_ops, len_seq in last_graph_missing
    }

    last_graph_included_dict = {
        f"{i},{j}": {
            "len_all_operations": len_ops,
            "len_graph_sequence": len_seq
        }
        for (i, j), len_ops, len_seq in last_graph_incl
    }

    # save results
    os.makedirs("../data/MUTAG/test", exist_ok=True)
    with open("../data/MUTAG/test/last_graph_missing_details.json", "w") as f:
        json.dump(last_graph_missing_dict, f, indent=2)

    with open("../data/MUTAG/test/last_graph_missing_diff.json", "w") as f:
        json.dump(diff_missing, f, indent=2)

    with open("../data/MUTAG/test/last_graph_included_details.json", "w") as f:
        json.dump(last_graph_included_dict, f, indent=2)

    with open("../data/MUTAG/test/last_graph_included_diff.json", "w") as f:
        json.dump(diff_included, f, indent=2)
