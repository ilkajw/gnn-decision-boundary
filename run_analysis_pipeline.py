import os
import subprocess
from sys import executable as PY
from pathlib import Path

from config import MODEL, DATASET_NAME, ANALYSIS_DIR


def run_step(script: str, msg: str):
    print(msg, flush=True)
    # Verify file exists before running
    if not Path(script).is_file():
        raise FileNotFoundError(f"Script not found: {script}")
    subprocess.run([PY, script], check=True)


if __name__ == "__main__":
    run_step("01_create_path_graphs.py",
             "Running graph creation. This may take some time...")

    run_step("02_assign_cumulative_costs.py",
             "Path graph sequences were created. Reconstructing and assigning cumulative costs...")

    run_step("03_train_model_on_original_dataset.py",
             f"Path graph sequences done. Training {MODEL} on {DATASET_NAME}...")

    run_step("04_classify_original_dataset.py",
             f"Model trained and best model saved. Classifying the {DATASET_NAME} graphs...")

    run_step("05_classify_path_graphs.py",
             f"{DATASET_NAME} graphs classified. Next, classifying the path graphs...")

    run_step("06_precalculations.py",
             "Path graphs classified. Running precalculations for analysis...")

    run_step("07_path_length_statistics.py",
             f"Precalculations done. Analysis starts. All results go to {ANALYSIS_DIR}. "
             "Calculating path length statistics for all paths...")

    run_step("08_path_length_statistics_per_num_flips.py",
             "Calculating path length statistics per number of flips...")

    run_step("09_num_flips_histograms.py",
             "Calculating number of flips distribution...")

    run_step("10_flip_order_distributions.py",
             "Calculating flip (order) distribution...")

    run_step("11_flip_statistics.py",
             "Calculating flip statistics...")

    plots_dir = os.path.join(ANALYSIS_DIR, "plots")
    run_step("12_plot_num_flips_histograms.py",
             f"Analysis done. Plotting results. Plots going to {plots_dir}...")

    run_step("13_plot_flip_order_distributions.py",
             "Plotting flip order distributions...")

    run_step("14_plot_flip_distributions_with_ops.py",
             "Plotting flip distributions with operations...")

    run_step("15_plot_operation_heatmaps.py",
             "Plotting operation heatmaps...")
