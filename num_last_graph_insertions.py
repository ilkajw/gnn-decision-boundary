import json

from config import DATASET_NAME, CORRECTLY_CLASSIFIED_ONLY
from index_sets_utils import build_index_set_cuts, cut_pairs

inserted_path = f"data/{DATASET_NAME}/test/last_graphs_inserted.json"
split_path = "model/best_split.json"
out_path = f"data/{DATASET_NAME}/analysis/last_graph_insertions.json"

# load index pairs where last graph was manually inserted
with open(inserted_path, "r") as f:
    insertions = json.load(f)

insert_set = {tuple(p) for p in insertions}
print(len(inserted_path))

# load index sets
cuts = build_index_set_cuts(
        dataset_name=DATASET_NAME,
        correctly_classified_only=CORRECTLY_CLASSIFIED_ONLY,
        split_path=split_path,
    )

keys = [
    "same_class_all", "same_class_0_all", "same_class_1_all", "diff_class_all",
    "same_train_train", "same_0_train_train", "same_1_train_train", "diff_train_train",
    "same_test_test",  "same_0_test_test",  "same_1_test_test",  "diff_test_test",
    "same_train_test", "same_0_train_test", "same_1_train_test", "diff_train_test",
]

# build dictionary of num paths with last graph inserted per idx set
insertions_per_idxset = {}

for key in keys:
    idx_set = cuts[key]
    insertions_per_idxset[key] = len(cut_pairs(insert_set, idx_set))

# save
with open(out_path, "w") as f:
    json.dump(insertions_per_idxset, f, indent=2)

print(insertions_per_idxset)
