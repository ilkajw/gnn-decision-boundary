from analyse_utils import count_paths_by_num_flips
from config import DATASET_NAME, CORRECTLY_CLASSIFIED_ONLY
from index_sets_utils import graph_index_pairs_same_class, graph_index_pairs_diff_class

if __name__ == "__main__":

    # todo: construct sets with helpers from index_sets_utils. filter for test/train too

    # ------------------------------- create index sets -------------------------------

    diff_class_pairs = graph_index_pairs_diff_class(
        dataset_name=DATASET_NAME,
        correctly_classified_only=CORRECTLY_CLASSIFIED_ONLY,
        save_path=f"data/{DATASET_NAME}/index_sets/{DATASET_NAME}_idx_pairs_diff_class.json")

    same_class_pairs, same_class_0_pairs, same_class_1_pairs = graph_index_pairs_same_class(
        dataset=DATASET_NAME,
        correctly_classified_only=CORRECTLY_CLASSIFIED_ONLY,
        save_dir=f"data/{DATASET_NAME}/index_sets/{DATASET_NAME}_idx_pairs")

    # ---------- count number of diff target/source class paths with k = 1, 3, 5... flips ----------------

    count_paths_by_num_flips(
        idx_pair_set=diff_class_pairs,
        flips_input_path=f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_changes_per_path.json",
        output_path=f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_num_paths_per_num_flips_diff_class.json",
        same_class=False)

    # ---------- count number of same target/source class paths with k = 0, 2, 4 ... flips ----------------
    # todo: check of the latter functions overwrite anything of the first
    # same
    count_paths_by_num_flips(
        idx_pair_set=same_class_pairs,
        flips_input_path=f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_changes_per_path.json",
        output_path=f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_num_paths_per_num_flips_same_class.json",
        same_class=True)

    # same class 1
    count_paths_by_num_flips(
        idx_pair_set=same_class_1_pairs,
        flips_input_path=f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_changes_per_path.json",
        output_path=f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_num_paths_per_num_flips_same_class_1.json",
        same_class=True)

    # same class 0
    count_paths_by_num_flips(
        idx_pair_set=same_class_0_pairs,
        flips_input_path=f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_changes_per_path.json",
        output_path=f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_num_paths_per_num_flips_same_class_0.json",
        same_class=True)