import os

from GAT import GAT
from GCN import GCN
from GraphSAGE import GraphSAGE

ROOT = './data_test'
DATASET_NAME = 'MUTAG'

# --- Set model architecture and parameters ---

MODEL = "GraphSAGE"  # <-------- define model here: "GAT" | "GCN" | "GraphSAGE

assert MODEL in {"GAT", "GCN", "GraphSAGE"}, \
    f"Invalid MODEL={MODEL!r}, must be 'GAT', 'GCN or 'GraphSAGE'."

MODEL_REGISTRY = {
    "GAT": (GAT, dict(hidden_channels=8, heads=8, dropout=0.2)),
    "GCN": (GCN, dict(hidden_channels=64, dropout=0.2)),
    "GraphSAGE": (GraphSAGE, dict(hidden_channels=64, dropout=0.2))
}

MODEL_CLS, MODEL_KWARGS = MODEL_REGISTRY[MODEL]
MODEL_DIR = os.path.join('models', DATASET_NAME, MODEL)

# --- Training parameters ---
K_FOLDS = 10
EPOCHS = 100
LEARNING_RATE = 0.005
BATCH_SIZE = 16


# ---- Analysis parameters ----

DISTANCE_MODE = "edit_step"  # 'cost' or 'edit_step' (as distance measure)

assert DISTANCE_MODE in {"cost", "edit_step"}, \
    f"Invalid DISTANCE_MODE={DISTANCE_MODE!r}, must be 'cost' or 'edit_step'."

CORRECTLY_CLASSIFIED_ONLY = True  # only consider paths with correctly classified source and target
FULLY_CONNECTED_ONLY = True  # only consider fully connected, hence plausible, edit path graphs


# ---- Paths ----

ANALYSIS_DIR = os.path.join(ROOT, DATASET_NAME, MODEL, 'analysis', f'by_{DISTANCE_MODE}',
                            f'correctly_classified={CORRECTLY_CLASSIFIED_ONLY}')

MODEL_INDEPENDENT_PRECALCULATIONS_DIR = os.path.join(ROOT, DATASET_NAME, "precalculations")

MODEL_DEPENDENT_PRECALCULATIONS_DIR = os.path.join(ROOT, DATASET_NAME, MODEL, "precalculations")

LEGACY_PYG_SEQ_DIR = os.path.join("data_control", DATASET_NAME, "pyg_edit_path_graphs")
PREDICTIONS_DIR = os.path.join(ROOT, DATASET_NAME, MODEL, "predictions")

# Define distance mode dependent precalculation files
if DISTANCE_MODE == "cost":
    DISTANCES_PATH = os.path.join(MODEL_INDEPENDENT_PRECALCULATIONS_DIR, f"{DATASET_NAME}_dist_per_path.json")
    FLIPS_PATH = os.path.join(MODEL_DEPENDENT_PRECALCULATIONS_DIR,
                              f"{DATASET_NAME}_{MODEL}_flip_occurrences_per_path_by_cost.json")
elif DISTANCE_MODE == "edit_step":
    DISTANCES_PATH = os.path.join(MODEL_INDEPENDENT_PRECALCULATIONS_DIR, f"{DATASET_NAME}_num_ops_per_path.json")
    FLIPS_PATH = os.path.join(MODEL_DEPENDENT_PRECALCULATIONS_DIR,
                              f"{DATASET_NAME}_{MODEL}_flip_occurrences_per_path_by_edit_step.json")


# ---- Synthetic dataset parameters ----
# Path progress at which to flip class label assigned to path graphs on paths between graphs of different class
FLIP_AT = 0.0
