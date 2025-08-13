from config import CORRECTLY_CLASSIFIED_ONLY
from index_sets_utils import *
from analyse_utils import *

if __name__ == "__main__":

    # test test
    stats_per_path_save_path = f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_changes_per_path_same_vs_diff_class.json"
    stats_per_step_save_path = f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_changes_per_edit_step_same_vs_diff_class.json"

    # get all index pairs of same and different classes
    diff_class_pairs = graph_index_pairs_diff_class(dataset_name=DATASET_NAME,
                                                    correctly_classified_only=CORRECTLY_CLASSIFIED_ONLY,
                                                    save_path=f"data/{DATASET_NAME}/index_sets/{DATASET_NAME}_idx_pairs_diff_class.json")

    same_class_pairs, same_class_0_pairs, same_class_1_pairs = graph_index_pairs_same_class(
        dataset=DATASET_NAME,
        correctly_classified_only=CORRECTLY_CLASSIFIED_ONLY,
        save_dir=f"data/{DATASET_NAME}/index_sets/{DATASET_NAME}_idx_pairs")

    if CORRECTLY_CLASSIFIED_ONLY:
        pass
        # todo: filter for correctly classified source, target,
        #       delete filtering from same/diff class idx set functions used above,
        #       encapsulate per path logic in function?

    # load pre-calculated predictions per path, prediction changes per path and number of changes per edit step

    with open(f"data/{DATASET_NAME}/predictions/{DATASET_NAME}_edit_path_predictions.json", "r") as f:
        predictions = json.load(f)

    with open(f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_changes_per_path.json") as f:
        changes_per_path_dict = json.load(f)

    with open(f"data/{DATASET_NAME}/analysis/{DATASET_NAME}_changes_per_edit_step.json") as f:
        num_changes_per_step_dict = json.load(f)

    # -----------------------------------PER PATH ANALYSIS-----------------------------------------

    # get lists of change count per path graph sequence for source and target
    # belonging to same or different classes

    same_class_changes = get_num_changes_all_paths(same_class_pairs, changes_per_path_dict)
    same_class_0_changes = get_num_changes_all_paths(same_class_0_pairs, changes_per_path_dict)
    same_class_1_changes = get_num_changes_all_paths(same_class_1_pairs, changes_per_path_dict)
    diff_class_changes = get_num_changes_all_paths(diff_class_pairs, changes_per_path_dict)

    #print(same_class_pairs)
    #print(same_class_changes)
    #print(same_class_0_changes)
    #print(same_class_1_changes)
    #print(diff_class_changes)

    # calculate statistics
    # todo: keep this
    stats_changes_per_path = {
        "same_class": {
            "num_paths": len(same_class_changes),
            "mean": float(np.mean(same_class_changes)) if same_class_changes else 0,
            "std": float(np.std(same_class_changes)) if same_class_changes else 0
        },
        "same_class_0": {
            "num_paths": len(same_class_0_changes),
            "mean": float(np.mean(same_class_0_changes)) if same_class_0_changes else 0,
            "std": float(np.std(same_class_0_changes)) if same_class_0_changes else 0
        },
        "same_class_1": {
            "num_paths": len(same_class_1_changes),
            "mean": float(np.mean(same_class_1_changes)) if same_class_1_changes else 0,
            "std": float(np.std(same_class_1_changes)) if same_class_1_changes else 0
        },
        "diff_class": {
            "num_paths": len(diff_class_changes),
            "mean": float(np.mean(diff_class_changes)) if diff_class_changes else 0,
            "std": float(np.std(diff_class_changes)) if diff_class_changes else 0
        }
    }

    # print(f"Statistics - Number of changes per path: \n {stats_changes_per_path}")

    # save
    os.makedirs(os.path.dirname(stats_per_path_save_path), exist_ok=True)
    with open(stats_per_path_save_path, "w") as f:
        json.dump(stats_changes_per_path, f, indent=2)

    # --------------------PER EDIT STEP ANALYSIS------------------------------------

    get_abs_flips_per_edit_step(
        idx_pairs_set=same_class_pairs,
        input_dir=f"data/{DATASET_NAME}/predictions/edit_path_graphs_with_predictions",
        output_dir=f"data/{DATASET_NAME}/analysis",
        output_fname=f"{DATASET_NAME}_changes_per_edit_step_same_class.json")

    get_abs_flips_per_edit_step(
        idx_pairs_set=same_class_0_pairs,
        input_dir=f"data/{DATASET_NAME}/predictions/edit_path_graphs_with_predictions",
        output_dir=f"data/{DATASET_NAME}/analysis",
        output_fname=f"{DATASET_NAME}_changes_per_edit_step_same_class_0.json")

    get_abs_flips_per_edit_step(
        idx_pairs_set=same_class_1_pairs,
        input_dir=f"data/{DATASET_NAME}/predictions/edit_path_graphs_with_predictions",
        output_dir=f"data/{DATASET_NAME}/analysis",
        output_fname=f"{DATASET_NAME}_changes_per_edit_step_same_class_1.json")

    get_abs_flips_per_edit_step(
        idx_pairs_set=diff_class_pairs,
        input_dir=f"data/{DATASET_NAME}/predictions/edit_path_graphs_with_predictions",
        output_dir=f"data/{DATASET_NAME}/analysis",
        output_fname=f"{DATASET_NAME}_changes_per_edit_step_diff_class.json")

    # -------------------------- PER DECILE ANALYSIS --------------------------------------------

    get_abs_flips_per_decile(
        idx_pairs_set=same_class_pairs,
        input_dir=f"data/{DATASET_NAME}/predictions/edit_path_graphs_with_predictions",
        output_dir=f"data/{DATASET_NAME}/analysis",
        output_fname=f"{DATASET_NAME}_changes_per_decile_same_class.json"
    )

    get_abs_flips_per_decile(
        idx_pairs_set=same_class_0_pairs,
        input_dir=f"data/{DATASET_NAME}/predictions/edit_path_graphs_with_predictions",
        output_dir=f"data/{DATASET_NAME}/analysis",
        output_fname=f"{DATASET_NAME}_changes_per_decile_same_class_0.json"
    )

    get_abs_flips_per_decile(
        idx_pairs_set=same_class_1_pairs,
        input_dir=f"data/{DATASET_NAME}/predictions/edit_path_graphs_with_predictions",
        output_dir=f"data/{DATASET_NAME}/analysis",
        output_fname=f"{DATASET_NAME}_changes_per_decile_same_class_1.json"
    )

    get_abs_flips_per_decile(
        idx_pairs_set=diff_class_pairs,
        input_dir=f"data/{DATASET_NAME}/predictions/edit_path_graphs_with_predictions",
        output_dir=f"data/{DATASET_NAME}/analysis",
        output_fname=f"{DATASET_NAME}_changes_per_decile_diff_class.json"
    )




