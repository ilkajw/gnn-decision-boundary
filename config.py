import os

from GAT import GAT
from GCN import GCN
from GraphSAGE import GraphSAGE

ROOT = './data_actual_best'
DATASET_NAME = 'MUTAG'

# --- set model and model params ---
MODEL = "GAT"  # or "GCN", "GraphSAGE
assert MODEL in {"GAT", "GCN", "GraphSAGE"}, \
    f"Invalid MODEL={MODEL!r}, must be 'GAT', 'GCN or 'GraphSAGE'."
MODEL_REGISTRY = {
    "GAT": (GAT, dict(hidden_channels=8, heads=8, dropout=0.2)),
    "GCN": (GCN, dict(hidden_channels=64, dropout=0.5, gat_heads=8)),
    "GraphSAGE": (GraphSAGE, dict(hidden_channels=64, dropout=0.5, gat_heads=8))
}
MODEL_CLS, MODEL_KWARGS = MODEL_REGISTRY[MODEL]
MODEL_DIR = f'models/{DATASET_NAME}/{MODEL}'

# --- training params ---
K_FOLDS = 10
EPOCHS = 100
LEARNING_RATE = 0.005
BATCH_SIZE = 16


# --- analysis params ---
DISTANCE_MODE = "cost"  # or 'edit_step' (distance measure)
assert DISTANCE_MODE in {"cost", "edit_step"}, \
    f"Invalid DISTANCE_MODE={DISTANCE_MODE!r}, must be 'cost' or 'edit_step'."

CORRECTLY_CLASSIFIED_ONLY = True  # only consider paths with correctly classified source and target
FULLY_CONNECTED_ONLY = True  # only consider fully connected, hence plausible, edit path graphs

# --- paths ---
ANALYSIS_DIR = f'{ROOT}/{DATASET_NAME}/{MODEL}/analysis/by_{DISTANCE_MODE}/'
LEGACY_PYG_SEQ_DIR = os.path.join("data_control", DATASET_NAME, "pyg_edit_path_graphs")
PREDICTIONS_DIR = f"{ROOT}/{DATASET_NAME}/{MODEL}/predictions"

# todo: put dist path and flips path in config

# --- synthetic dataset params ---
FLIP_AT = 0.5  # path progress
