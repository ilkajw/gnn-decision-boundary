"""
Train a GNN on a TUDataset using k-fold cross-validation.

This script loads a dataset from PyTorch Geometric's TUDataset (path defined by
`ROOT`/`DATASET_NAME`), constructs the model class `MODEL_CLS` with parameters
`MODEL_KWARGS`, and trains it with the project's cross-validation training
utility `train_model_kcv`.

Outputs:
  - Trained model saved to: MODEL_DIR / f"{DATASET_NAME}_{MODEL}_model.pt"
  - Best split info saved to: MODEL_DIR / f"{DATASET_NAME}_{MODEL}_best_split.json"
  - Training log saved to: MODEL_DIR / f"{DATASET_NAME}_{MODEL}_train_log.json"

Config / requirements:
  - Expects `config.py` to expose ROOT, DATASET_NAME, MODEL, MODEL_CLS, MODEL_KWARGS, MODEL_DIR.
  - Requires torch-geometric and the utilities in training_utils.
"""

import os
from torch_geometric.datasets import TUDataset

from config import ROOT, DATASET_NAME, MODEL, MODEL_CLS, MODEL_KWARGS, MODEL_DIR
from training_utils import train_model_kcv

# --- Set inputs ---
output_dir = MODEL_DIR
model_fname = f"{DATASET_NAME}_{MODEL}_model.pt"
split_fname = f"{DATASET_NAME}_{MODEL}_best_split.json"
log_fname = f"{DATASET_NAME}_{MODEL}_train_log.json"


# --- Run ---
if __name__ == "__main__":

    dataset = TUDataset(root=ROOT, name=DATASET_NAME)
    os.makedirs(output_dir, exist_ok=True)

    train_model_kcv(
        model_cls=MODEL_CLS,
        model_kwargs=MODEL_KWARGS,
        dataset=dataset,
        output_dir=output_dir,
        model_fname=model_fname,
        split_fname=split_fname,
        log_fname=log_fname
    )

