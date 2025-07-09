from predict import *
from edit_path_graphs import *
from train import train_and_choose_model
from analyse import *


if __name__ == "__main__":

    dataset = TUDataset(root=ROOT, name=DATASET_NAME)

    # todo: this is only temporary. add suitable logic to structure the pipeline/access parts of the pipeline
    train_model = False
    dataset_predict = False
    calc_edit_path_graphs = False
    edit_path_predict = False
    add_meta_data = False

    if train_model:
        train_and_choose_model(dataset=dataset,
                               output_dir="model",
                               model_fname="model.pt",
                               split_fname="best_split.json",
                               log_fname="log.json")
    if dataset_predict:
        dataset_predictions(dataset_name=DATASET_NAME,
                            output_dir=f"data/{DATASET_NAME}/predictions/",
                            output_fname=f"{DATASET_NAME}_predictions.json",
                            model_path="model/model.pt")

    if calc_edit_path_graphs:
        generate_all_edit_path_graphs(db_name=DATASET_NAME,
                                      seed=42,
                                      data_dir=f"external/pg_gnn_edit_paths/example_paths_{DATASET_NAME}",
                                      output_dir=f"data/{DATASET_NAME}/pyg_edit_path_graphs",
                                      fully_connected_only=FULLY_CONNECTED_ONLY)
    if edit_path_predict:
        pred_dict = edit_path_predictions(dataset_name=DATASET_NAME,
                                          model_path="model/model.pt",
                                          input_dir=f"data/{DATASET_NAME}/pyg_edit_path_graphs",
                                          output_dir=f"data/{DATASET_NAME}/predictions",
                                          output_fname=f"{DATASET_NAME}_edit_path_predictions.json")

        add_metadata_to_edit_path_predictions(pred_dict_path=f"data/{DATASET_NAME}/predictions/{DATASET_NAME}_edit_path_predictions.json",
                                              base_pred_path=f"data/{DATASET_NAME}/predictions/{DATASET_NAME}_predictions.json",
                                              split_path=f"model/best_split.json",
                                              output_path=f"data/{DATASET_NAME}/predictions/{DATASET_NAME}_edit_path_predictions_metadata.json")

    count_class_changes_per_edit_step(input_dir=f"data/{DATASET_NAME}/predictions/edit_path_graphs_with_predictions",
                                      output_dir=f"data/{DATASET_NAME}/predictions",
                                      output_fname=f"{DATASET_NAME}_changes_per_edit_step.json",
                                      verbose=True)

    get_class_change_steps_per_pair(input_dir=f"data/{DATASET_NAME}/predictions/edit_path_graphs_with_predictions",
                                    output_dir=f"data/{DATASET_NAME}/predictions",
                                    output_fname=f"{DATASET_NAME}_changes_per_path.json",
                                    verbose=True)
