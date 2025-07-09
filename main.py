from predict import *
from edit_path_graphs import *
from train import train_and_choose_model
from analyse import *
from index_sets import *

if __name__ == "__main__":
    dataset_name = 'MUTAG'
    dataset = TUDataset(root=ROOT, name=dataset_name)
    training =False
    dataset_predict = False
    edit_path_graph = False
    edit_path_predict = False
    add_meta_data = False

    # todo: add logic to check if data exists, else calculate
    # todo: where to get edit distance for pairs from? needed to analyse where class flips happen
    if training:
        train_and_choose_model(dataset=dataset,
                               output_dir="model",
                               model_fname="model.pt",
                               split_fname="best_split.json",
                               log_fname="log.json")
    if dataset_predict:
        dataset_predictions(dataset_name=dataset_name,
                            output_dir=f"data/{dataset_name}/predictions/",
                            output_fname=f"{dataset_name}_predictions.json",
                            model_path="model/model.pt")

    if edit_path_graph:
        generate_and_save_all_edit_path_graphs(db_name=dataset_name,
                                               seed=42,
                                               data_dir=f"external/pg_gnn_edit_paths/example_paths_{dataset_name}",
                                               output_dir=f"data/{dataset_name}/pyg_edit_path_graphs",
                                               fully_connected_only=True)
    if edit_path_predict:
        pred_dict = edit_path_predictions(dataset_name=dataset_name,
                                          model_path="model/model.pt",
                                          input_dir=f"data/{dataset_name}/pyg_edit_path_graphs",
                                          output_dir=f"data/{dataset_name}/predictions",
                                          output_fname=f"{dataset_name}_edit_path_predictions.json")

    if add_meta_data:
        add_metadata_to_edit_path_predictions(pred_dict_path=f"data/{dataset_name}/predictions/{dataset_name}_edit_path_predictions.json",
                                              base_pred_path=f"data/{dataset_name}/predictions/{dataset_name}_predictions.json",
                                              split_path=f"model/best_split.json",
                                              output_path=f"data/{dataset_name}/predictions/{dataset_name}_edit_path_predictions_metadata.json")

    count_class_changes_per_edit_step(input_dir=f"data/{dataset_name}/predictions/edit_path_graphs_with_predictions",
                                      output_dir=f"data/{dataset_name}/predictions",
                                      output_fname=f"{dataset_name}_changes_per_edit_step.json",
                                      verbose=True)

    get_class_change_steps_per_pair(input_dir=f"data/{dataset_name}/predictions/edit_path_graphs_with_predictions",
                                    output_dir=f"data/{dataset_name}/predictions",
                                    output_fname=f"{dataset_name}_changes_per_path.json",
                                    verbose=True)

    graph_index_pairs_diff_class(dataset_name=dataset_name,
                                 correctly_classified_only=True,
                                 save_path=f"data/{dataset_name}/index_sets/{dataset_name}_idx_pairs_diff_class.json")

    graph_index_pairs_same_class(dataset_name=dataset_name,
                                 correctly_classified_only=True,
                                 save_path=f"data/{dataset_name}/index_sets/{dataset_name}_idx_pairs_same_class.json")
