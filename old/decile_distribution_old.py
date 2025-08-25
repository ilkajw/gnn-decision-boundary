from old.analyse_utils import flip_distribution_over_deciles_by_indexset
from config import DATASET_NAME, CORRECTLY_CLASSIFIED_ONLY
from index_sets_utils import graph_index_pairs_diff_class, graph_index_pairs_same_class

if __name__ == "__main__":

    # todo: construct sets with helpers from index_sets_utils, including same/diff + test/train

    # -------------------------- create index sets -------------------------------------

    diff_class_pairs = graph_index_pairs_diff_class(dataset_name=DATASET_NAME,
                                                    correctly_classified_only=CORRECTLY_CLASSIFIED_ONLY,
                                                    save_path=f"data/{DATASET_NAME}/index_sets/{DATASET_NAME}_idx_pairs_diff_class.json")

    same_class_pairs, same_class_0_pairs, same_class_1_pairs = graph_index_pairs_same_class(
        dataset=DATASET_NAME,
        correctly_classified_only=CORRECTLY_CLASSIFIED_ONLY,
        save_dir=f"data/{DATASET_NAME}/index_sets/{DATASET_NAME}_idx_pairs")

    # ----------------- calculate flip distributions per index set ---------------------------

    flip_distribution_over_deciles_by_indexset(
        idx_pair_set=same_class_pairs,
        dist_input_path=f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_dist_per_pair.json",
        flips_input_path=f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_changes_per_path.json",
        output_path=f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_rel_flips_per_decile_same_class.json")

    flip_distribution_over_deciles_by_indexset(
        idx_pair_set=same_class_0_pairs,
        dist_input_path=f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_dist_per_pair.json",
        flips_input_path=f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_changes_per_path.json",
        output_path=f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_rel_flips_per_decile_same_class_0.json")

    flip_distribution_over_deciles_by_indexset(
        idx_pair_set=same_class_1_pairs,
        dist_input_path=f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_dist_per_pair.json",
        flips_input_path=f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_changes_per_path.json",
        output_path=f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_rel_flips_per_decile_same_class_1.json")

    flip_distribution_over_deciles_by_indexset(
        idx_pair_set=diff_class_pairs,
        dist_input_path=f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_dist_per_pair.json",
        flips_input_path=f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_changes_per_path.json",
        output_path=f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_rel_flips_per_decile_diff_class.json")