from plotting_utils import plot_num_flips_histograms_from_files, plot_decile_distributions_from_files
from config import DATASET_NAME

plot_num_flips_histograms_from_files(
    label_to_file={
        "train_train": f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_num_paths_per_num_flips_train_train.json",
        "test_test": f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_num_paths_per_num_flips_test_test.json",
        "train_test": f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_num_paths_per_num_flips_train_test.json"
    },
    normalize=False,
    title=f"{DATASET_NAME}: #Flips per Path (Counts)",
    save_path=f"data/{DATASET_NAME}/figures/{DATASET_NAME}_num_flips_counts.png"
)

plot_num_flips_histograms_from_files(
    label_to_file={
        "train_train": f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_num_paths_per_num_flips_train_train.json",
        "test_test": f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_num_paths_per_num_flips_test_test.json",
        "train_test": f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_num_paths_per_num_flips_train_test.json"
    },
    normalize=True,
    title=f"{DATASET_NAME}: #Flips per Path (Proportions)",
    save_path=f"data/{DATASET_NAME}/figures/{DATASET_NAME}_num_flips_proportions.png"
)


plot_decile_distributions_from_files(
    label_to_file={
        "same_class": f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_rel_flips_per_decile_same_class.json",
        "diff_class": f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_rel_flips_per_decile_diff_class.json"
    },
    title=f"{DATASET_NAME}: Relative Flip Distribution by Decile",
    save_path=f"data/{DATASET_NAME}/figures/{DATASET_NAME}_rel_flips_deciles.png"
)