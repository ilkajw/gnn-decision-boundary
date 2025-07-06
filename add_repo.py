import sys
import os

# Add submodule root to Python path
submodule_path = os.path.abspath("external/pg_gnn_edit_paths")
if submodule_path not in sys.path:
    sys.path.insert(0, submodule_path)
