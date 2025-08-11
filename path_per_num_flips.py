from analyse_utils import *
from config import DATASET_NAME

if __name__ == "__main__":

    # calculate the distribution of flips per set of paths having k = 1, 2, 3, ... flips

    paths_per_num_flips = get_rel_flips_per_decile_by_k(
        max_k = 8,
        dist_input_path=f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_dist_per_pair.json",
        flips_input_path=f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_changes_per_path.json",
        output_dir=f"data/{DATASET_NAME}/analysis/",
        output_prefix=f"{DATASET_NAME}_flip_distribution_per_num_flips")



