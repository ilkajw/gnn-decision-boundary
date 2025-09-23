import subprocess

if __name__ == "__main__":

    print("Running graph creation. This may take some time...")
    subprocess.run(["python", "01_create_path_graphs.py"])

    print("Path graph sequences were created. We're reconstructing and assigning their cumulative costs now. "
          "This will take a moment...")
    subprocess.run(["python", "02_assign_cumulative_costs.py"])

