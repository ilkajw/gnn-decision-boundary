from analyse_utils import get_distance_per_pair, get_flip_steps_per_pair
from config import DATASET_NAME

if __name__ == "__main__":

    get_distance_per_pair(input_path=f"external/pg_gnn_edit_paths/example_paths_{DATASET_NAME}",
                          output_path=f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_dist_per_pair.json")

    get_flip_steps_per_pair(input_dir=f"data/{DATASET_NAME}/predictions/edit_path_graphs_with_predictions",
                            output_dir=f"data/{DATASET_NAME}/analysis",
                            output_fname=f"{DATASET_NAME}_changes_per_path.json")
