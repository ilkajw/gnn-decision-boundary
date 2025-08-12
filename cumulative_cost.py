import json
import os
import torch

from config import DATASET_NAME
from external.pg_gnn_edit_paths.utils.io import load_edit_paths_from_file


def seq_filename(i, j, it, root_dir):
    # matches your naming: g{i}_to_g{j}_it{it}_graph_sequence.pt
    return os.path.join(root_dir, f"g{i}_to_g{j}_it{it}_graph_sequence.pt")


def build_cumulative_costs_from_operations(
    db_name,
    file_path,
    node_del_cost=1,
    node_ins_cost=1,
    edge_del_cost=1,
    edge_ins_cost=1,
    node_subst_cost=0,
    edge_subst_cost=0
):
    """
    Build a dictionary mapping (i, j) -> [cumulative cost per step]
    from the edit paths loaded via load_edit_paths_from_file.

    - Costs are accumulated in the order of `all_operations`.
    - Iteration number in the edit path is ignored; we key only by (i, j).
    """
    edit_paths_map = load_edit_paths_from_file(db_name, file_path)
    if edit_paths_map is None:
        raise FileNotFoundError(f"No edit path file found for {db_name} in {file_path}")

    cum_costs = {}

    for (i, j), paths in edit_paths_map.items():
        if not paths:
            continue

        ep = paths[0]   # currently only one path per pair, potentially adjust for more paths from other datasets
        cumulative = [0.0]
        cost = 0.0

        for op, val in ep.all_operations:
            if op == "remove_node":
                cost += node_del_cost
            elif op == "add_node":
                cost += node_ins_cost
            elif op == "remove_edge":
                cost += edge_del_cost
            elif op == "add_edge":
                cost += edge_ins_cost
            elif op == "relabel_node":
                cost += node_subst_cost
            elif op == "relabel_edge":
                cost += edge_subst_cost
            # append the running total
            cumulative.append(cost)

        cum_costs[(i, j)] = cumulative

    return cum_costs


def seq_filename(i, j, root_dir):
    # g{i}_to_g{j}_it*_graph_sequence.pt   (iteration ignored, we try both orders)
    # If your files do not have "_itX_", adjust this pattern accordingly.
    # For simplicity, we keep using the direct filename pattern you had earlier:
    return os.path.join(root_dir, f"g{i}_to_g{j}_it1_graph_sequence.pt")  # or your exact pattern


def _align_costs_to_seq(seq, costs):
    """
    Make cost vector length match seq_len.
    - If seq_len == len(costs)+1: append last cost (extra terminal graph).
    - If seq_len == len(costs)-1: drop last cost.
    - If larger diffs: pad with last cost or trim, and warn.
    """
    c = list(costs)
    seq_len = len(seq)
    cost_len = len(c)

    if seq_len == cost_len:
        return c, None
    if seq_len == cost_len + 1:
        # todo: how to handle best?
        # extra terminal graph, repeat final cost
        return c + [c[-1]], "padded(+1)"
    else:
        print(f"Larger mismatch between len sequence and len cum cost. Check!")
        return None, None


def add_cumulative_cost_to_pyg_sequence_metadata(
        cum_costs,  # dict[(i,j)] -> [cumulative costs]
        root_dir,
        cost_field: str = "cumulative_cost",
        out_dir: str | None = None):

    os.makedirs(out_dir or root_dir, exist_ok=True)

    # loop over all paths
    for (i, j), costs in cum_costs.items():

        # load graph sequence from file
        in_path = seq_filename(i, j, root_dir)
        if not os.path.exists(in_path):
            alt = seq_filename(j, i, root_dir)
            if os.path.exists(alt):
                in_path = alt
            else:
                print(f"[warn] seq missing for {(i, j)}: {in_path}")
                continue
        sequence = torch.load(in_path, map_location="cpu", weights_only=False)

        # align costs to sequence length to handle approximate paths
        costs_aligned, note = _align_costs_to_seq(costs, sequence)
        # catch falsy cost list len
        if costs_aligned is None:
            print(f"[warn] mismatch between len graph sequence ({len({sequence})} and len cum cost list ({len(costs)})"
                  f" for {i}, {j}. Returning without saving")
            return
        if note:
            print(f"[info] align {(i,j)}: {note}  (seq_len={len(sequence)}, costs_len={len(costs)})")

        # update each graph
        for g in sequence:
            # original edit step
            step_idx = int(getattr(g, "edit_step", 0))
            # clamp into valid range (in case of out-of-bounds)
            step_idx = max(0, min(step_idx, len(costs_aligned) - 1))
            # add new attribute
            setattr(g, cost_field, float(costs_aligned[step_idx]))

        # save update
        out_path = in_path if out_dir is None else os.path.join(out_dir, os.path.basename(in_path))
        torch.save(sequence, out_path)

        print(f"[pyg] updated: {out_path}")


def add_cumulative_cost_to_predictions_metadata(
    pred_json_path,
    cum_costs,                      # dict[(i,j)] -> [cumulative costs]
    cost_field: str = "cumulative_cost",
    out_path: str | None = None,
):
    # load predictions with meta-data
    with open(pred_json_path, "r") as f:
        entries = json.load(f)

    updated = [],
    for e in entries:
        i = int(e["source_idx"])
        j = int(e["target_idx"])
        step_idx = int(e.get("edit_step", 0))

        key = (i, j)
        if key not in cum_costs and (j, i) in cum_costs:
            key = (j, i)
        if key not in cum_costs:
            print(f"pair {i}, {j} not in cum cost. returning without further saving")
            return

        # get cumulative cost list for pair
        c = cum_costs[key]
        # happens for approx paths with last graph inserted. currently last cost will be assigned
        if step_idx >= len(c):
            step_idx -= 1  # todo: how to handle?
        # assign cum cost to entry
        e[cost_field] = float(c[step_idx])
        updated.append(e)

    # save updated predictions
    if out_path is None:
        out_path = pred_json_path.replace(".json", f"_WITH_{cost_field.upper()}.json")
    with open(out_path, "w") as f:
        json.dump(updated, f, indent=2)

    print(f"[predictions] wrote: {out_path}  (missing_pairs={missing}, clamped_steps={oob})")


if __name__ == "__main__":

    cum_costs = build_cumulative_costs_from_operations(
        db_name=DATASET_NAME,
        file_path=f"external/pg_gnn_edit_paths/example_paths_{DATASET_NAME}"
    )

    add_cumulative_cost_to_predictions_metadata(
        pred_json_path=fr"data\{DATASET_NAME}\predictions\{DATASET_NAME}_edit_path_predictions_metadata.json",
        cum_costs=cum_costs,
        cost_field="cumulative_cost",
    )

    add_cumulative_cost_to_pyg_sequence_metadata(
        cum_costs=cum_costs,
        root_dir=fr"data\{DATASET_NAME}\predictions\edit_path_graphs_with_predictions",
        cost_field="cumulative_cost",
    )
