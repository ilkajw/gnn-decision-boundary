import os
import re
import json
import torch
from torch.serialization import add_safe_globals
from torch_geometric.data import Data

from config import DATASET_NAME, MODEL, PREDICTIONS_DIR
from external.pg_gnn_edit_paths.utils.io import load_edit_paths_from_file

# --- Set input paths ---
edit_path_ops_dir = f"external/pg_gnn_edit_paths/example_paths_{DATASET_NAME}"
path_predictions_json = f"{PREDICTIONS_DIR}/{DATASET_NAME}_{MODEL}_edit_path_predictions.json"

# --- Set output path ---
out_file = path_predictions_json  # Overwrite


# --- Helpers ---
# not used (yet)
def add_operations_to_pyg_sequences(
    db_name,                      # dict[(i,j)] -> [cumulative costs]
    ops_file_dir,
    seq_dir,                       # directory with *.pt sequences
    add_field_name: str = "cumulative_cost",
    out_dir: str | None = None,  # if None, overwrite original sequences
):
    """
    Walks 'root_dir', finds files matching: g{i}_to_g{j}_it{ANY}_graph_sequence.pt,
    reads the sequence, aligns cumulative costs for (i,j), adds attribute 'cumulative_cost' to each graph
    and writes an updated sequence to 'out_dir' or overwrites input directory if 'out_dir' is None.
    """

    edit_paths_map = load_edit_paths_from_file(db_name, ops_file_dir)
    if edit_paths_map is None:
        raise FileNotFoundError(f"No edit path file found for {db_name} in {ops_file_dir}")

    # Compile filename pattern
    pattern = re.compile(r"g(\d+)_to_g(\d+)_it\d+_graph_sequence\.pt")

    out_dir = out_dir or (seq_dir)
    os.makedirs(out_dir, exist_ok=True)
    add_safe_globals([Data])

    processed = 0
    skipped_no_entry = 0
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
        if key not in edit_paths_map and (j, i) in edit_paths_map:
            key = (j, i)
        if key not in edit_paths_map:
            skipped_no_entry += 1
            print(f"[ERROR] no .paths entry found for pair {key} (file: {fname}). skipping.")
            continue
        paths = edit_paths_map[key]
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

        # Assign cumulative cost per graph
        for g in seq:
            path = paths[0]
            step_idx = int(getattr(g, "edit_step"))
            if step_idx == 0:
                g.operation = None
            ops_list_idx = step_idx-1
            if ops_list_idx >= len(path["all_operations"]):
                if ops_list_idx - len(path["all_operations"]) > 1:
                    raise ValueError(f"edit_step {step_idx} (equals ops_list_idx {ops_list_idx}) oob for pair {(i, j)} "
                    f"(len(path.all_operations)={len(path['all_operations'])}, len(sequence)={len(seq)}). "
                    f"Difference is {ops_list_idx - len(path['all_operations'])} > 1.)")

                else:
                    print(f"[INFO] edit_step {step_idx} (equals ops_list_idx {ops_list_idx}) oob for pair {(i, j)} "
                        f"(len(path.all_operations)={len(path['all_operations'])}, len(sequence)={len(seq)}). "
                          f"Difference should be 1 at maximum."
                    )
                    g.operation = None

            else:
                g.operation = path['all_operations'][step_idx-1]

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
        f"skipped_no_entry={skipped_no_entry}, "
        f"skipped_no_match={skipped_no_match}, "
        f"missing_files={missing_files}"
    )


def add_operations_to_path_predictions_json(
        db_name,
        ops_file_dir,
        path_pred_json_path,
        out_path: str | None = None,  # if None overwrite
):

    # Load predictions JSON
    with open(path_pred_json_path, "r") as f:
        entries = json.load(f)

    edit_paths_map = load_edit_paths_from_file(db_name, ops_file_dir)
    if edit_paths_map is None:
        raise FileNotFoundError(f"No edit path file found for {db_name} in {ops_file_dir}")

    updated = []

    # Loop over all graph entries from JSON
    for e in entries:

        # Get source and target index of current graph
        i = int(e["source_idx"])
        j = int(e["target_idx"])
        step_idx = int(e.get("edit_step"))

        paths = edit_paths_map[(i, j)] if (i, j) in edit_paths_map else (
            edit_paths_map[(j, i)] if (j, i) in edit_paths_map else
            (_ for _ in ()).throw(KeyError(f"pair {(i, j)} not in .paths"))
        )

        path = paths[0]
        all_ops = path["all_operations"]
        num_ops = len(all_ops)

        if step_idx > num_ops+1:
            raise ValueError(f"edit_step {step_idx} out of allowed range [0..{num_ops+1}] for pair {(i,j)}")

        e["operation"] = None if step_idx in (0, num_ops+1) else path['all_operations'][step_idx-1]
        updated.append(e)

    if out_path is None:
        # Overwrite input dir. was: pred_json_path.replace(".json", f"_WITH_{add_field_name.upper()}.json")
        out_path = path_pred_json_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(updated, f, indent=2)


# ---- Run ----
if __name__ == "__main__":

    # Fail fast if inputs missing
    for p in [edit_path_ops_dir, path_predictions_json]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing input directory: {p}")

    os.makedirs(os.dirname(out_file), exist_ok=True)

    add_operations_to_path_predictions_json(
        db_name=DATASET_NAME,
        ops_file_dir=edit_path_ops_dir,
        path_pred_json_path=path_predictions_json,
        out_path=out_file,
    )
