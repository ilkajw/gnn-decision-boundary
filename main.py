from predict import *
from edit_path_graphs import *
from train import train_and_choose_model

if __name__ == "__main__":
    dataset_name = 'MUTAG'
    dataset = TUDataset(root=ROOT, name=dataset_name)

    train_and_choose_model(dataset=dataset,
                           output_dir="model",
                           model_fname="model.pt",
                           split_fname="best_split.json",
                           log_fname="log.json")

    dataset_predictions(dataset_name=dataset_name,
                        output_dir=f"data/{dataset_name}/predictions/",
                        output_fname=f"{dataset_name}_predictions.json",
                        model_path="model/model.pt")

    generate_and_save_all_edit_path_graphs(db_name=dataset_name,
                                           seed=42,
                                           data_dir=f"external/pg_gnn_edit_paths/example_paths_{dataset_name}",
                                           output_dir=f"data/{dataset_name}/pyg_edit_path_graphs",
                                           fully_connected_only=True)

    pred_dict = edit_path_predictions(dataset_name=dataset_name,
                                      model_path="model/model.pt",
                                      input_dir=f"data/{dataset_name}/pyg_edit_path_graphs",
                                      output_dir=f"data/{dataset_name}/predictions",
                                      output_fname=f"{dataset_name}_edit_path_predictions.json")

    add_metadata_to_edit_path_predictions(
        pred_dict_path=f"data/{dataset_name}/predictions/{dataset_name}_edit_path_predictions.json",
        base_pred_path=f"data/{dataset_name}/predictions/{dataset_name}_predictions.json",
        split_path=f"model/best_split.json",
        output_path=f"data/{dataset_name}/predictions/{dataset_name}_edit_path_predictions_metadata.json")
