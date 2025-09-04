import os
import re
import json
import torch
from torch.serialization import add_safe_globals
from torch_geometric.data import Data

from config import DATASET_NAME, MODEL, MODEL_DIR, PREDICTIONS_DIR
from external.pg_gnn_edit_paths.utils.io import load_edit_paths_from_file

# TODO: delete cum cost addition here, then full file ist not needed. check again

# --- Set input paths ---

edit_path_ops_dir = f"external/pg_gnn_edit_paths/example_paths_{DATASET_NAME}"

pyg_seq_with_preds_dir = f"{PREDICTIONS_DIR}/edit_path_graphs_with_predictions"

base_pred_dict_fname = f"{DATASET_NAME}_{MODEL}_predictions.json"
base_pred_dict_path = os.path.join(PREDICTIONS_DIR, base_pred_dict_fname)

# Will be overwritten with enriched attr
path_pred_dict_fname = f"{DATASET_NAME}_{MODEL}_edit_path_predictions.json"
path_pred_dict_path = os.path.join(PREDICTIONS_DIR, path_pred_dict_fname)

split_path = f"{MODEL_DIR}/{DATASET_NAME}_{MODEL}_best_split.json"


# --- Set output path ---
seq_out_dir = f"{PREDICTIONS_DIR}/edit_path_graphs_with_predictions_CUMULATIVE_COST"


# --- Helpers ---

# TODO: only info on source_idx and target_idx are used later. delete rest?
def add_attrs_to_path_preds_dict(path_pred_json_path, base_pred_json_path, split_path, output_path):
    """
    Enriches dictionary entries of edit path predictions with additional metadata to help later analysis:
    - true class labels of source and target
    - whether source/target are in training split
    - whether source/target were classified correctly

    Args:
        :param path_pred_json_path:
        :param split_path: Path to saved train, test split
        :param base_pred_json_path: Path to original MUTAG predictions file
        :param output_path: Path to file where enriched dictionary is saved

    Returns:
        list: Enriched prediction dictionaries
    """
    # load path graph predictions
    with open(path_pred_json_path, "r", encoding="utf-8") as f:
        pred_dict = json.load(f)

    # load predictions on org graphs
    with open(base_pred_json_path, "r", encoding="utf-8") as f:
        base_preds = json.load(f)

    # load train, test split
    with open(split_path, "r") as f:
        split = json.load(f)

    # Add info to entries:
    # train vs. test split,
    # classes of source & target,
    # correct classification of source & train
    for entry in pred_dict:

        i = str(entry["source_idx"])
        j = str(entry["target_idx"])

        entry["source_in_train"] = int(i) in split["train_idx"]
        entry["target_in_train"] = int(j) in split["train_idx"]

        entry["correct_source"] = base_preds[i]["correct"]
        entry["correct_target"] = base_preds[j]["correct"]

        entry["source_class"] = base_preds[i]["true_label"]
        entry["target_class"] = base_preds[j]["true_label"]

    with open(output_path, "w") as f:
        json.dump(pred_dict, f, indent=2)
    print(f"[info] saved enriched predictions to {output_path}")


def build_cum_costs_from_ops(
    db_name,
    ops_file_dir,
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
    edit_paths_map = load_edit_paths_from_file(db_name, ops_file_dir)
    if edit_paths_map is None:
        raise FileNotFoundError(f"No edit path file found for {db_name} in {ops_file_dir}")

    cum_costs = {}

    for (i, j), paths in edit_paths_map.items():

        if not paths:
            print(f"[warn] no path found for {i}, {j}")
            continue

        ep = paths[0]  # currently only one path per pair todo: potentially adjust for more paths from other datasets
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
    - If seq_len == len(costs)+1: append last cost (inserted terminal graph).
    - If seq_len == len(costs)-1: drop last cost.
    """
    c = list(costs)
    last_graph = seq[-1]
    max_edit_step = getattr(last_graph, 'edit_step')
    cost_len = len(c)

    # todo: how to handle best?
    # Handle paths where target graph inserted,
    # assign additional cost 0.0 for inserted target graph to underestimate true cost
    if max_edit_step == cost_len:
        return c + [c[-1]], "padded cost (+1)"

    if max_edit_step > cost_len:
        return c, f"[ERROR] max_edit_step {max_edit_step} > len_costs ({cost_len}+1={cost_len+1})."

    else:
        return c, None


def add_cum_cost_to_pyg_seq(
    cum_costs,                      # dict[(i,j)] -> [cumulative costs]
    seq_dir,                       # directory with *.pt sequences
    add_field_name: str = "cumulative_cost",
    out_dir: str | None = None,  # if None, overwrite original sequences
):
    """
    Walks 'root_dir', finds files matching: g{i}_to_g{j}_it{ANY}_graph_sequence.pt,
    reads the sequence, aligns cumulative costs for (i,j), adds attribute 'cumulative_cost' to each graph
    and writes an updated sequence to 'out_dir' or overwrites input directory if 'out_dir' is None.
    """

    # Compile filename pattern
    pattern = re.compile(r"g(\d+)_to_g(\d+)_it\d+_graph_sequence\.pt")

    out_dir = out_dir or (seq_dir)
    os.makedirs(out_dir, exist_ok=True)
    add_safe_globals([Data])

    processed = 0
    skipped_no_costs = 0
    skipped_no_match = 0
    missing_files = 0

    for fname in sorted(os.listdir(seq_dir)):

        if not fname.endswith(".pt"):
            continue

        m = pattern.fullmatch(fname)
        if not m:
            skipped_no_match += 1
            print(f"[info] not a graph-seq filename: {fname}")
            continue

        i, j = int(m.group(1)), int(m.group(2))

        # Find costs for (i,j) or (j,i)
        key = (i, j)
        if key not in cum_costs and (j, i) in cum_costs:
            key = (j, i)
        if key not in cum_costs:
            skipped_no_costs += 1
            print(f"[ERROR] no cum_costs for pair {(i, j)} (file: {fname}). skipping.")
            continue

        # Load sequence object
        in_path = os.path.join(seq_dir, fname)
        try:
            obj = torch.load(in_path, map_location="cpu", weights_only=False)
        except FileNotFoundError:
            missing_files += 1
            print(f"[error] missing file: {in_path}")
            continue

        # Support dict-wrapped or raw list
        if isinstance(obj, dict):
            if "graphs" in obj:
                seq, wrapper_key = obj["graphs"], "graphs"
            elif "sequence" in obj:
                seq, wrapper_key = obj["sequence"], "sequence"
            else:
                # Assume dict itself is the sequence-like
                seq, wrapper_key = obj, None
        else:
            seq, wrapper_key = obj, None

        # Align costs to sequence length to handle paths with target graph insertion
        costs = cum_costs[key]
        costs_aligned, note = _align_costs_to_seq(seq=seq, costs=costs)
        if note:
            print(f"[info] align {(i, j)}: {note} (seq_len={len(seq)}, costs_len={len(costs)})")

        # Assign cumulative cost per graph
        for g in seq:
            step_idx = int(getattr(g, "edit_step"))
            # Check for oob edit step
            if step_idx < 0 or step_idx >= len(costs_aligned):
                raise ValueError(
                    f"[ERROR] edit_step {step_idx} out of range for pair {(i, j)} "
                    f"(len(costs_aligned)={len(costs_aligned)}, len(sequence)={len(seq)})"
                )
            #
            setattr(g, add_field_name, float(costs_aligned[step_idx]))

        # Save to output dir with same filename
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


def add_cum_cost_to_path_preds_json(
    path_pred_json_path,
    cum_costs,                      # dict[(i,j)] -> [cumulative costs]
    add_field_name: str = "cumulative_cost",
    out_path: str | None = None,  # if None overwrite
):
    # Load predictions JSON
    with open(path_pred_json_path, "r") as f:
        entries = json.load(f)

    updated = []

    # Loop over all graph entries from JSON
    for e in entries:

        # Get source and target index of current graph
        i = int(e["source_idx"])
        j = int(e["target_idx"])

        key = (i, j)
        if key not in cum_costs and (j, i) in cum_costs:
            key = (j, i)
        if key not in cum_costs:
            print(f"[error] pair {(i, j)} not in cum_costs dict. skipping entry.")
            continue

        # Get cumulative cost list for pair
        c = cum_costs[key]

        # Get edit step of current graph
        step_idx = int(e.get("edit_step"))

        # todo: how to handle best?
        # If graph is inserted target graph, assume additional cost 0.0 to underestimate true cost,
        # hence assign cost of previous graph
        if step_idx >= len(c):
            step_idx -= 1

        # Add cum cost as attribute
        e[add_field_name] = float(c[step_idx])
        updated.append(e)

    if out_path is None:
        # Overwrite input dir. was: pred_json_path.replace(".json", f"_WITH_{add_field_name.upper()}.json")
        out_path = path_pred_json_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(updated, f, indent=2)


# --- Run ---
if __name__ == "__main__":

    # Fail fast if inputs missing
    for p in [edit_path_ops_dir, base_pred_dict_path, path_pred_dict_path, pyg_seq_with_preds_dir, split_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing input directory: {p}")

    os.makedirs(seq_out_dir, exist_ok=True)

    add_attrs_to_path_preds_dict(
            path_pred_json_path=path_pred_dict_path,
            base_pred_json_path=base_pred_dict_path,
            split_path=split_path,
            output_path=path_pred_dict_path,  # overwrite
    )

    cum_costs = build_cum_costs_from_ops(
        db_name=DATASET_NAME,
        ops_file_dir=edit_path_ops_dir
    )

    add_cum_cost_to_path_preds_json(
        path_pred_json_path=path_pred_dict_path,
        cum_costs=cum_costs,
        add_field_name="cumulative_cost",
        out_path=path_pred_dict_path,  # overwrite
    )

    add_cum_cost_to_pyg_seq(
        cum_costs=cum_costs,
        seq_dir=pyg_seq_with_preds_dir,
        add_field_name="cumulative_cost",
        out_path=seq_out_dir,  # write to separate dir. if None -> overwrite
    )
