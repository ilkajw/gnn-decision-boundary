import subprocess
import os

from config import MODEL, DATASET_NAME, ANALYSIS_DIR

if __name__ == "__main__":

    print("Running graph creation. This may take some time...")
    subprocess.run(["python", "01_create_path_graphs.py"])

    print("Path graph sequences were created. We're reconstructing and assigning their cumulative costs now. "
          "This will take a moment...")
    subprocess.run(["python", "02_assign_cumulative_costs.py"])

    print(f"Path graph sequences done. Training {MODEL} on {DATASET_NAME}...")
    subprocess.run(["python", "03_train_model_on_original_dataset.py"])

    print(f"Model trained and best model saved. Classifying the {DATASET_NAME} graphs...")
    subprocess.run(["python", "04_classify_original_dataset.py"])

    print(f"{DATASET_NAME} graphs classified. Next, classifying path graphs...")
    subprocess.run(["python", "05_classify_path_graphs.py"])

    print(f"Path graphs classified. Running precalculations for analysis...")
    subprocess.run(["python", "06_precalculations.py"])

    print(f"Precalculations done. Ananlysis starts. All results go to {ANALYSIS_DIR}"
          f"Calculating path length statistics for all paths...")
    subprocess.run(["python", "07_path_length_stats.py"])
    subprocess.run(["python", "08_path_length_stats_per_num_flips.py"])

    print(f"Calculating number of flips distribution...")
    subprocess.run(["python", "09_num_flips_histograms.py"])

    print(f"Calculating flip (order) distribution...")
    subprocess.run(["python", "10_flip_order_distributions.py"])

    print(f"Calculating flip statistics...")
    subprocess.run(["python", "11_flip_stats.py"])

    print(f"Analysis done. Plotting results. Plots going to {os.path.join(ANALYSIS_DIR, 'plots')}...")
    subprocess.run(["python"], "12_plot_num_flips_histograms.py")
    subprocess.run(["python"], "13_plot_flip_order_distributions.py")
    subprocess.run(["python"], "14_plot_flip_distributions_with_ops.py")
    subprocess.run(["python"], "15_plot_operation_heatmaps.py")




