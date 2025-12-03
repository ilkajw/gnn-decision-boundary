import subprocess
from sys import executable as PY
from pathlib import Path


def run_step(script: str, msg: str):
    print(msg, flush=True)
    if not Path(script).is_file():
        raise FileNotFoundError(f"Script not found: {script}")
    subprocess.run([PY, script], check=True)


if __name__ == "__main__":
    run_step("01_create_path_graphs.py",
             "Running graph creation. This may take some time...")

    run_step("02_assign_cum_costs.py",
             "Path graph sequences were created. "
             "Reconstructing and assigning cumulative costs. This will take a moment...")
