import os
import sys

# if current directory is not in sys.path, add it
utils_path = os.path.join(os.path.dirname(__file__), "..")
if utils_path not in sys.path:
    sys.path.append(utils_path)
