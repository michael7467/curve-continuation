"""Lets the scripts run from the project folder without installing the package (pip install -e . works as well)."""

import os
import sys
from pathlib import Path

# numpy's OpenBLAS can fail with "Memory allocation still failed" when it starts many threads; one is enough here
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
