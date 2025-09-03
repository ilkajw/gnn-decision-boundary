import os, json, torch
from collections import Counter
from torch_geometric.datasets import TUDataset
from config import DATASET_NAME

MERGED = f"data_control/{DATASET_NAME}/processed/{DATASET_NAME}_merged_dataset_tagged.pt"
OUT_JSON = f"data_control/{DATASET_NAME}/analysis/{DATASET_NAME}_class_shares.json"

def print_stats(name, c):
    total = sum(c.values())
    c0, c1 = c.get(0, 0), c.get(1, 0)
    p = lambda x: (100*x/total) if total else 0.0
    print(f"\n{name}: total={total}\n  class 0: {c0} ({p(c0):.2f}%)\n  class 1: {c1} ({p(c1):.2f}%)")
    return {
        "total": int(total),
        "class_0": int(c0),
        "class_1": int(c1),
        "share_0": float(p(c0)),
        "share_1": float(p(c1)),
    }

# --- Original MUTAG (scalar y per graph) ---
orig = TUDataset(root=f"data_control/{DATASET_NAME}/pyg", name="MUTAG")
c_orig = Counter(int(g.y.view(-1)[0].item()) for g in orig if hasattr(g, "y"))
orig_stats = print_stats("Original MUTAG", c_orig)

# --- Merged dataset saved as (data, slices) ---
data, slices = torch.load(MERGED, map_location="cpu", weights_only=False)
y = data.y.view(-1).float()
ptr = slices["y"]  # graph boundaries
c_merged = Counter()
for i in range(ptr.numel() - 1):
    yi = y[ptr[i]:ptr[i+1]]               # labels for graph i (length 1 or many)
    lbl = int((yi.mean() >= 0.5).item())  # binarize to 0/1
    c_merged[lbl] += 1
merged_stats = print_stats("Merged dataset", c_merged)

# --- Save to JSON ---
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
payload = {
    "meta": {
        "dataset": DATASET_NAME,
        "merged_path": MERGED,
        "binarize_rule": "mean(y) >= 0.5",
    },
    "original": orig_stats,
    "merged": merged_stats,
}
with open(OUT_JSON, "w") as f:
    json.dump(payload, f, indent=2)
print(f"\nSaved JSON → {OUT_JSON}")
