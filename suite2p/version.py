"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

from importlib.metadata import PackageNotFoundError, version as package_version
import sys
from platform import python_version
import torch

try:
    version = package_version("suite2p")
except PackageNotFoundError:
    version = "unknown"

try:
    cellpose_version = package_version("cellpose")
except PackageNotFoundError:
    cellpose_version = "not installed"

version_str = f"""
suite2p version: \t{version}
cellpose version:\t{cellpose_version}
platform:       \t{sys.platform}
python version: \t{python_version()}
torch version:  \t{torch.__version__}"""
