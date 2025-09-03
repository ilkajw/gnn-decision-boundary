import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":16:8"
os.environ["PYTHONHASHSEED"] = "42"

from predict_utils import *
from old.edit_path_graphs_utils import *
from training_utils import train_model_kcv
from old.analyse_utils import *


if __name__ == "__main__":

    dataset = TUDataset(root=ROOT, name=DATASET_NAME)

    train_model = False
    predict_dataset = False
    gen_edit_path_graphs = True
    predict_edit_paths = True
    add_meta_data_to_path_preds = True

    if train_model:
        train_model_kcv(dataset=dataset,
                        output_dir="../model_control",
                        model_fname="model.pt",
                        split_fname="best_split.json",
                        log_fname="log.json")
    if predict_dataset:
        dataset_predictions(dataset_name=DATASET_NAME,
                            output_dir=f"data_control/{DATASET_NAME}/predictions/",
                            output_fname=f"{DATASET_NAME}_predictions.json",
                            model_path="../model_control/model.pt")

    if gen_edit_path_graphs:
        generate_edit_path_graphs(
            db_name=DATASET_NAME,
            seed=42,
            data_dir=f"external/pg_gnn_edit_paths/example_paths_{DATASET_NAME}",
            output_dir=f"data_control/{DATASET_NAME}/pyg_edit_path_graphs",
            fully_connected_only=FULLY_CONNECTED_ONLY)

    if predict_edit_paths:
        pred_dict = edit_path_predictions(
            dataset_name=DATASET_NAME,
            model_path="../model_control/model.pt",
            input_dir=f"data_control/{DATASET_NAME}/pyg_edit_path_graphs",
            output_dir=f"data_control/{DATASET_NAME}/predictions",
            output_fname=f"{DATASET_NAME}_edit_path_predictions.json")

    if add_meta_data_to_path_preds:
        add_metadata_to_path_preds_dict(
            pred_dict_path=f"data_control/{DATASET_NAME}/predictions/{DATASET_NAME}_edit_path_predictions.json",
            base_pred_path=f"data_control/{DATASET_NAME}/predictions/{DATASET_NAME}_predictions.json",
            split_path=f"../model_control/best_split.json",
            output_path=f"data_control/{DATASET_NAME}/predictions/{DATASET_NAME}_edit_path_predictions_metadata.json")

