from predict_utils import *
from edit_path_graphs_utils import *
from training_utils import train_and_choose_model
from analyse_utils import *


if __name__ == "__main__":

    dataset = TUDataset(root=ROOT, name=DATASET_NAME)

    # todo: this is only temporary. add suitable logic to access parts of the pipeline

    train_model = False
    predict_dataset = False
    gen_edit_path_graphs = True
    predict_edit_paths = False
    add_meta_data = False

    if train_model:
        train_and_choose_model(dataset=dataset,
                               output_dir="model",
                               model_fname="model.pt",
                               split_fname="best_split.json",
                               log_fname="log.json")
    if predict_dataset:
        dataset_predictions(dataset_name=DATASET_NAME,
                            output_dir=f"data/{DATASET_NAME}/predictions/",
                            output_fname=f"{DATASET_NAME}_predictions.json",
                            model_path="model/model.pt")

    if gen_edit_path_graphs:
        generate_all_edit_path_graphs(db_name=DATASET_NAME,
                                      seed=42,
                                      data_dir=f"external/pg_gnn_edit_paths/example_paths_{DATASET_NAME}",
                                      output_dir=f"data/{DATASET_NAME}/pyg_edit_path_graphs",
                                      fully_connected_only=False)
    if predict_edit_paths:
        pred_dict = edit_path_predictions(dataset_name=DATASET_NAME,
                                          model_path="model/model.pt",
                                          input_dir=f"data/{DATASET_NAME}/pyg_edit_path_graphs",
                                          output_dir=f"data/{DATASET_NAME}/predictions",
                                          output_fname=f"{DATASET_NAME}_edit_path_predictions.json")

    if add_meta_data:
        add_metadata_to_edit_path_predictions(pred_dict_path=f"data/{DATASET_NAME}/predictions/{DATASET_NAME}_edit_path_predictions.json",
                                              base_pred_path=f"data/{DATASET_NAME}/predictions/{DATASET_NAME}_predictions.json",
                                              split_path=f"model/best_split.json",
                                              output_path=f"data/{DATASET_NAME}/predictions/{DATASET_NAME}_edit_path_predictions_metadata.json")

