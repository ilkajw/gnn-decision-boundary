import os

from GAT import GAT
from GCN import GCN
from GraphSAGE import GraphSAGE

ROOT = './data_actual_best'
DATASET_NAME = 'MUTAG'

# --- set model and model params ---

MODEL = "GAT"  # <-------- define model here: "GAT" | "GCN" | "GraphSAGE

assert MODEL in {"GAT", "GCN", "GraphSAGE"}, \
    f"Invalid MODEL={MODEL!r}, must be 'GAT', 'GCN or 'GraphSAGE'."
MODEL_REGISTRY = {
    "GAT": (GAT, dict(hidden_channels=8, heads=8, dropout=0.2)),
    "GCN": (GCN, dict(hidden_channels=64, dropout=0.2)),
    "GraphSAGE": (GraphSAGE, dict(hidden_channels=64, dropout=0.2))
}
MODEL_CLS, MODEL_KWARGS = MODEL_REGISTRY[MODEL]
MODEL_DIR = f'models/{DATASET_NAME}/{MODEL}'

# --- training params ---
K_FOLDS = 10
EPOCHS = 100
LEARNING_RATE = 0.005
BATCH_SIZE = 16


# ---- Analysis definitions ----

DISTANCE_MODE = "edit_step"  # or 'edit_step' (distance measure)
assert DISTANCE_MODE in {"cost", "edit_step"}, \
    f"Invalid DISTANCE_MODE={DISTANCE_MODE!r}, must be 'cost' or 'edit_step'."

CORRECTLY_CLASSIFIED_ONLY = True  # only consider paths with correctly classified source and target
FULLY_CONNECTED_ONLY = True  # only consider fully connected, hence plausible, edit path graphs


# --- Paths ---

ANALYSIS_DIR = f'{ROOT}/{DATASET_NAME}/{MODEL}/analysis/by_{DISTANCE_MODE}/' \
               f'correctly_classified={CORRECTLY_CLASSIFIED_ONLY}'

MODEL_DEPENDENT_PRECALCULATIONS_DIR = f'{ROOT}/{DATASET_NAME}/{MODEL}/precalculations'
MODEL_INDEPENDENT_PRECALCULATIONS_DIR = f'{ROOT}/{DATASET_NAME}/precalculations'

LEGACY_PYG_SEQ_DIR = os.path.join("data_control", DATASET_NAME, "pyg_edit_path_graphs")
PREDICTIONS_DIR = f"{ROOT}/{DATASET_NAME}/{MODEL}/predictions"

# Pick the right precalculation files
# TODO: use in files, not used yet
if DISTANCE_MODE == "cost":
    DISTANCES_PATH = os.path.join(MODEL_INDEPENDENT_PRECALCULATIONS_DIR, f"{DATASET_NAME}_dist_per_path.json")
elif DISTANCE_MODE == "edit_step":
    DISTANCES_PATH = os.path.join(MODEL_INDEPENDENT_PRECALCULATIONS_DIR, f"{DATASET_NAME}_num_ops_per_path.json")

# todo: put dist path and flips path in config


# ---- Synthetic dataset parameters ----

FLIP_AT = 0.9  # path progress
