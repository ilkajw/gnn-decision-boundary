import json
import os
import re
import torch

from config import DATASET_NAME
from external.pg_gnn_edit_paths.utils.io import load_edit_paths_from_file


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


def _align_costs_to_seq(seq, costs):
    """
    Make cost vector length match seq_len.
    - If seq_len == len(costs)+1: append last cost (extra terminal graph).
    - If seq_len == len(costs)-1: drop last cost.
    """
    c = list(costs)
    last_graph = seq[-1]
    max_edit_step = getattr(last_graph, 'edit_step')
    cost_len = len(c)

    # todo: how to handle best?
    # handle paths where target graph included, currently repeats final cost
    if max_edit_step == cost_len:
        return c + [c[-1]], "padded cost (+1)"
    # this should not happen
    if max_edit_step > cost_len:
        return c, f"[error] max_edit_step {max_edit_step} > len_costs ({cost_len}+1={cost_len+1})."
    else:
        return c, None


def add_cumulative_cost_to_pyg_sequence_metadata(
    cum_costs,                      # dict[(i,j)] -> [cumulative costs]
    root_dir,                       # directory with *.pt sequences
    cost_field: str = "cumulative_cost",
    out_dir: str | None = None,
):
    """
    Walks 'root_dir', finds files matching: g{i}_to_g{j}_it{ANY}_graph_sequence.pt,
    reads the sequence, aligns cumulative costs for (i,j), adds attribute 'cumulative_cost' to each graph
    and writes an updated sequence copy to 'out_dir'.
    """
    # compile filename pattern once
    pattern = re.compile(r"g(\d+)_to_g(\d+)_it\d+_graph_sequence\.pt")

    # choose a safe output directory
    out_dir = out_dir or (root_dir + "_CUMULATIVE_COST")
    os.makedirs(out_dir, exist_ok=True)

    processed = 0
    skipped_no_costs = 0
    skipped_no_match = 0
    missing_files = 0

    for fname in os.listdir(root_dir):
        if not fname.endswith(".pt"):
            continue

        m = pattern.fullmatch(fname)
        if not m:
            skipped_no_match += 1
            print(f"[skip] not a graph-seq filename: {fname}")
            continue

        i, j = int(m.group(1)), int(m.group(2))

        in_path = os.path.join(root_dir, fname)

        # find costs for (i,j) or (j,i)
        key = (i, j)
        if key not in cum_costs and (j, i) in cum_costs:
            key = (j, i)
        if key not in cum_costs:
            skipped_no_costs += 1
            print(f"[warn] no cum_costs for pair {(i, j)} (file: {fname}); skipping.")
            continue

        # load sequence object
        try:
            obj = torch.load(in_path, map_location="cpu", weights_only=False)
        except FileNotFoundError:
            missing_files += 1
            print(f"[warn] missing file (race condition?): {in_path}")
            continue

        # support dict-wrapped or raw list
        if isinstance(obj, dict):
            if "graphs" in obj:
                seq, wrapper_key = obj["graphs"], "graphs"
            elif "sequence" in obj:
                seq, wrapper_key = obj["sequence"], "sequence"
            else:
                # assume dict itself is the sequence-like
                seq, wrapper_key = obj, None
        else:
            seq, wrapper_key = obj, None

        # align costs to seq length to handle approximate paths
        costs = cum_costs[key]
        costs_aligned, note = _align_costs_to_seq(seq=seq, costs=costs)
        if note:
            print(f"[info] align {(i, j)}: {note} (seq_len={len(seq)}, costs_len={len(costs)})")

        # assign cumulative cost per graph (keep existing edit_step index)
        for g in seq:
            step_idx = int(getattr(g, "edit_step"))
            # check for oob edit step
            if step_idx < 0 or step_idx >= len(costs_aligned):
                raise ValueError(
                    f"[error] edit_step {step_idx} out of range for pair {(i, j)} "
                    f"(len(costs_aligned)={len(costs_aligned)}, seq_len={len(seq)})"
                )
            #
            setattr(g, cost_field, float(costs_aligned[step_idx]))
            # also store explicit index for later reference
            # todo: this is for safety only, should not be necessary
            if not hasattr(g, "edit_step_index"):
                setattr(g, "edit_step_index", int(step_idx))

        # save to separate output dir with same filename
        out_path = os.path.join(out_dir, fname)
        if isinstance(obj, dict) and wrapper_key is not None:
            obj[wrapper_key] = seq
            torch.save(obj, out_path)
        else:
            torch.save(seq, out_path)

        processed += 1
        print(f"[pyg] updated: {out_path}")

    print(
        f"[summary] processed={processed}, "
        f"skipped_no_costs={skipped_no_costs}, "
        f"skipped_no_match={skipped_no_match}, "
        f"missing_files={missing_files}"
    )


def add_cumulative_cost_to_predictions_metadata(
    pred_json_path,
    cum_costs,                      # dict[(i,j)] -> [cumulative costs]
    cost_field: str = "cumulative_cost",
    out_path: str | None = None,
):
    # load predictions with meta-data
    with open(pred_json_path, "r") as f:
        entries = json.load(f)

    updated = []
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
        # happens for approx paths with last graph inserted. currently last cost will be assigned for inserted graph
        if step_idx >= len(c):
            step_idx -= 1  # todo: how to handle best?
        # assign cum cost to entry
        e[cost_field] = float(c[step_idx])
        updated.append(e)

    # save updated predictions
    if out_path is None:
        out_path = pred_json_path.replace(".json", f"_WITH_{cost_field.upper()}.json")
    with open(out_path, "w") as f:
        json.dump(updated, f, indent=2)


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
