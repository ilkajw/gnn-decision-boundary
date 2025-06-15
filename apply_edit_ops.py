import networkx as nx
import copy
import os
import pickle


def apply_edit_operations(g1, g2, path, graphs="graphs"):

    # create dir to save graphs
    output_dir = "edit_path_graphs"
    os.makedirs(output_dir, exist_ok=True)

    # copy of original g1 for changes
    current_graph = copy.deepcopy(g1)

    # add attributes on changes for later readability
    nx.set_node_attributes(current_graph, 'unchanged', 'status')

    # init counter which equals ged
    step_counter = 1

    # array to save all edit path graphs between g1 and g2
    graph_sequence = []

    for match in path:
        # new snapshot of current graph
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
            new_node = f'v{v}'
            g_step.add_node(new_node)
            g_step.nodes[new_node]['status'] = 'inserted'
            if 'label' in g2.nodes[v]:
                g_step.nodes[new_node]['label'] = g2.nodes[v]['label']

        file_path = os.path.join(output_dir, f"{graphs}_sequence.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(graph_sequence, f)
        print(f"Saved step {step_counter} to {file_path}")

        # add ged=step_counter to graph's meta data
        g_step.graph['edit_step'] = step_counter

        # save in list of edit path graphs between g1, g2
        graph_sequence.append(g_step)

        current_graph = g_step
        step_counter += 1

    return
