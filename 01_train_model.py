import os
from torch_geometric.datasets import TUDataset

from config import ROOT, DATASET_NAME, MODEL, MODEL_CLS, MODEL_KWARGS, MODEL_DIR
from training_utils import train_model_kcv

# --- config ---
output_dir = MODEL_DIR
model_fname = f"{DATASET_NAME}_{MODEL}_model.pt"
split_fname = f"{DATASET_NAME}_{MODEL}_best_split.json"
log_fname = f"{DATASET_NAME}_{MODEL}_train_log.json"


# --- run ---
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

