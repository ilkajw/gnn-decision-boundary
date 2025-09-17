import os
import re
import json
import torch
from torch.serialization import add_safe_globals
from torch_geometric.data import Data

from config import ROOT, DATASET_NAME, MODEL, PREDICTIONS_DIR
from external.pg_gnn_edit_paths.utils.io import load_edit_paths_from_file

# --- Set input paths ---
edit_path_ops_dir = os.path.join("../external", "pg_gnn_edit_paths", f"example_paths_{DATASET_NAME}")
path_predictions_json = os.path.join(PREDICTIONS_DIR, f"{DATASET_NAME}_{MODEL}_edit_path_predictions.json")
sequence_directory = os.path.join(ROOT, DATASET_NAME, 'pyg_edit_path_graphs')

# --- Set output path ---
out_file = path_predictions_json  # Overwrite


# --- Helpers ---
def add_operations_to_pyg_sequences(
    dataset_name: str,
    paths_file_directory: str,
    sequence_directory: str,
    output_directory: str | None = None,  # if None, overwrite originals
    verbose: bool = True
):
    """
    For each file g{i}_to_g{j}_it{k}_graph_sequence.pt in 'sequence_directory', load the
    corresponding edit path object (matching (i,j) and iteration k), and annotate every
    graph in the sequence with the operation that produced it:
      - graph 0 (source): g.operation = "start_graph"
      - graph t>0: the (t-1)-th operation from ep.all_operations, or "target_graph_insertion"
        iff the sequence required an appended target (i.e., t-1 == len(ep.all_operations)).
    Writes updated sequences to 'output_directory' or overwrites inputs if None.
    STRICT: no direction swaps, no wrapper formats, no fallbacks.
    """

    # Load (i, j) -> list[EditPath] where each has .iteration, .all_operations
    edit_paths_map = load_edit_paths_from_file(dataset_name, paths_file_directory)
    if edit_paths_map is None:
        raise FileNotFoundError(f"No edit path file found for {dataset_name} in {paths_file_directory}")

    # g{i}_to_g{j}_it{k}_graph_sequence.pt  (capture i, j, k)
    pattern = re.compile(r"^g(\d+)_to_g(\d+)_it(\d+)_graph_sequence\.pt$")

    output_directory = output_directory or sequence_directory
    os.makedirs(output_directory, exist_ok=True)

    # Needed to torch.load older PyG Data objects
    add_safe_globals([Data])

    processed = 0
    skipped_no_match = 0

    for fname in sorted(os.listdir(sequence_directory)):
        if not fname.endswith(".pt"):
            continue

        m = pattern.fullmatch(fname)
        if not m:
            skipped_no_match += 1
            raise RuntimeError(f"Unexpected filename format (strict): {fname}")

        i, j, k = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        key = (i, j)

        if key not in edit_paths_map:
            raise KeyError(f"No edit paths entry for pair {(i, j)} (file: {fname})")

        # Find the path object for the specific iteration k. Not needed yet, to be future compatible
        eps = edit_paths_map[key]
        ep = None
        for cand in eps:
            # cand is an object with attribute 'iteration'
            if getattr(cand, "iteration", None) == k:
                ep = cand
                break
        if ep is None:
            raise KeyError(f"No path with iteration={k} for pair {(i, j)} (file: {fname})")

        in_path = os.path.join(sequence_directory, fname)
        seq = torch.load(in_path, map_location="cpu", weights_only=False)

        # Strict sequence structure: a list/tuple of torch_geometric.data.Data
        if not isinstance(seq, (list, tuple)) or len(seq) == 0:
            raise RuntimeError(f"Loaded sequence is empty or not a list/tuple: {fname}")
        if not all(isinstance(g, Data) for g in seq):
            raise TypeError(f"Sequence contains non-Data entries: {fname}")

        num_ops = len(getattr(ep, "all_operations"))
        # Expected: len(seq) == num_ops + 1 (source included) OR +2 if target insertion occurred
        #if not (len(seq) == num_ops + 1 or len(seq) == num_ops + 2):
         #   raise RuntimeError(
          #      f"Inconsistent sequence length in {fname}: len(seq)={len(seq)}, len(all_operations)={num_ops}"
           # )

        # Annotate each graph with the operation that produced it
        for g in seq:
            if not hasattr(g, "edit_step"):
                raise AttributeError(f"'edit_step' missing in a graph of {fname}")
            step_idx = int(getattr(g, "edit_step"))

            if step_idx == 0:
                g.operation = "start_graph"
                continue

            op_idx = step_idx - 1
            if op_idx < num_ops:
                # Operation that produced this graph
                g.operation = ep.all_operations[op_idx][0]
            elif op_idx == num_ops:
                # The sequence needed to append the target graph (one extra step beyond recorded ops)
                g.operation = "target_graph_insertion"
            else:
                # More than one step beyond known ops -> invalid
                raise IndexError(
                    f"edit_step {step_idx} (op_idx={op_idx}) exceeds allowed bound "
                    f"(num_ops={num_ops}) in {fname}"
                )

        # Save (overwrite or to output dir). We do not support wrappers; be strict.
        out_path = os.path.join(output_directory, fname)
        torch.save(seq, out_path)
        processed += 1
        if verbose:
            print(f"[pyg] updated: {out_path}")

    print(f"[summary] processed={processed}, skipped_no_match={skipped_no_match}")


def add_operations_to_path_predictions_json(
    db_name: str,
    ops_file_dir: str,
    path_pred_json_path: str,
    out_path: str | None = None,   # if None overwrite
):
    # Load predictions JSON
    with open(path_pred_json_path, "r") as f:
        entries = json.load(f)
    if not isinstance(entries, list):
        raise TypeError("Predictions JSON must be a list of entries.")

    # Load edit paths map: dict[(i,j)] -> list[EditPath]
    edit_paths_map = load_edit_paths_from_file(db_name, ops_file_dir)
    if edit_paths_map is None:
        raise FileNotFoundError(f"No edit path file found for {db_name} in {ops_file_dir}")

    updated = []

    for e in entries:
        # Required fields
        try:
            i = int(e["source_idx"])
            j = int(e["target_idx"])
            step_idx = int(e["edit_step"])
        except KeyError as ke:
            raise KeyError(f"Missing required key in entry: {ke}; entry={e}") from ke
        except ValueError as ve:
            raise ValueError(f"Non-integer source_idx/target_idx/edit_step in entry: {e}") from ve

        # Strict: require iteration in JSON and exact (i,j) key
        if "iteration" not in e:
            raise KeyError(f"'iteration' missing in entry for pair {(i, j)}; entry={e}")
        iteration = int(e["iteration"])

        key = (i, j)
        if key not in edit_paths_map:
            raise KeyError(f"Pair {(i, j)} not present in .paths (no direction swapping).")

        # Select the exact path object by iteration
        ep = None
        for cand in edit_paths_map[key]:
            if getattr(cand, "iteration", None) == iteration:
                ep = cand
                break
        if ep is None:
            raise KeyError(f"No path with iteration={iteration} for pair {(i, j)}.")

        # Operations are on the path object
        if not hasattr(ep, "all_operations"):
            raise AttributeError(f"EditPath object missing 'all_operations' for {(i, j)} iter {iteration}.")
        all_ops = ep.all_operations
        for idx, op in enumerate(all_ops):
            if not (isinstance(op, (list, tuple)) and len(op) >= 1 and isinstance(op[0], str)):
                raise TypeError(
                    f"Invalid op at index {idx} for pair {(i, j)} iter {iteration}: got {op!r}"
                )
        num_ops = len(all_ops)

        # Allowed steps:
        #  0           -> source graph
        #  1..num_ops  -> graphs produced by recorded operations
        #  num_ops+1   -> appended target (if present)
        if step_idx < 0 or step_idx > num_ops + 1:
            raise ValueError(
                f"edit_step {step_idx} out of allowed range [0..{num_ops+1}] "
                f"for pair {(i, j)}, iteration {iteration}"
            )

        if step_idx == 0:
            op_label = "start_graph"
        elif step_idx <= num_ops:
            op_label = all_ops[step_idx - 1][0]
        else:  # step_idx == num_ops + 1
            op_label = "target_graph_insertion"

        # Write the operation back (don’t mutate other fields)
        e["operation"] = op_label
        updated.append(e)

    out_path = out_path or path_pred_json_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(updated, f, indent=2)


# ---- Run ----
if __name__ == "__main__":

    # Fail fast if inputs missing
    for p in [edit_path_ops_dir, sequence_directory, path_predictions_json]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing input directory: {p}")

    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    add_operations_to_pyg_sequences(dataset_name=DATASET_NAME,
                                    paths_file_directory=edit_path_ops_dir,
                                    sequence_directory=sequence_directory,
                                    )
    add_operations_to_path_predictions_json(
        db_name=DATASET_NAME,
        ops_file_dir=edit_path_ops_dir,
        path_pred_json_path=path_predictions_json,
        out_path=out_file,
    )
