import json

from config import DATASET_NAME
from utils.io import load_edit_paths_from_file

#
edit_paths = load_edit_paths_from_file(db_name=DATASET_NAME,
                                       file_path=f"external/pg_gnn_edit_paths/example_paths_{DATASET_NAME}")

with open(f"data_control/{DATASET_NAME}/analysis/no_intermediates/"
          f"{DATASET_NAME}_no_intermediate_graphs_at_dataset_build.json", "r") as f:
    no_intermediates_at_dataset_build = json.load(f)

with open(f"data_control/{DATASET_NAME}/analysis/no_intermediates/"
          f"{DATASET_NAME}_no_intermediate_graphs_at_graph_seq_creation.json", "r") as f:
    no_intermediates_at_graph_seq_creation = json.load(f)

no_ops = [
    (i, j)
    for (i, j), paths in edit_paths.items()
    for ep in paths
    if len(ep.all_operations) == 0
]

no_intermediates = list({
    tuple(pair) for pair in no_intermediates_at_dataset_build
} - set(no_ops))

with open(f"data_control/{DATASET_NAME}/analysis/no_intermediates/{DATASET_NAME}_no_intermediates_but_ops.json", "w") as f:
    json.dump(no_intermediates, f, indent=2)

print(f"# no intermediate graphs at dataset build: {len(no_intermediates_at_dataset_build)}")
print(f"# no intermediate graphs at graph sequence creation: {len(no_intermediates_at_dataset_build)}")
print(f"# no operations: {len(no_ops)}")