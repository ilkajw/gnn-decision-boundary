from analyse_utils import *
from config import DATASET_NAME

if __name__ == "__main__":

    # todo: add index set discrimination?

    # retrieve data according to distance mode
    if DISTANCE_MODE == "cost":
        dist_path = f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_dist_per_path.json"
        flips_path = f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_flip_occurrences_per_path_by_cum_cost.json"
    else:
        dist_path = f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_num_ops_per_path.json"
        flips_path = f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_flip_occurrences_per_path_by_edit_step.json"

    # calculate the distribution of flips per set of paths having k = 1, 2, 3, ... flips
    flip_distribution_over_deciles_by_num_flips(
        max_num_flips=8,
        dist_input_path=dist_path,
        flips_input_path=flips_path,
        output_path=f"data/{DATASET_NAME}/analysis/num_paths_per_num_flips/{DISTANCE_MODE}/"
                    f"{DATASET_NAME}_flip_distribution_per_num_flips_by_{DISTANCE_MODE}.json")
