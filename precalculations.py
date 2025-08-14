from analyse_utils import get_distance_per_path, get_num_ops_per_path, \
    flip_occurrences_per_path_edit_step, flip_occurrences_per_path_cum_cost
from config import DATASET_NAME

if __name__ == "__main__":

    get_distance_per_path(input_path=f"external/pg_gnn_edit_paths/example_paths_{DATASET_NAME}",
                          output_path=f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_dist_per_path.json")

    get_num_ops_per_path(input_path=f"external/pg_gnn_edit_paths/example_paths_{DATASET_NAME}",
                         output_path=f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_num_ops_per_path.json")

    flip_occurrences_per_path_edit_step(
        input_dir=f"data/{DATASET_NAME}/predictions/edit_path_graphs_with_predictions",
        output_dir=f"data/{DATASET_NAME}/analysis",
        output_fname=f"{DATASET_NAME}_flip_occurrences_per_path_by_edit_step.json")

    flip_occurrences_per_path_cum_cost(
        input_dir=f"data/{DATASET_NAME}/predictions/edit_path_graphs_with_predictions_CUMULATIVE_COST",
        output_dir=f"data/{DATASET_NAME}/analysis",
        output_fname=f"{DATASET_NAME}_flip_occurrences_per_path_by_cost.json")
