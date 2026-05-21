from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DATA_FILES = [
    PROJECT_ROOT / "train.csv",
    PROJECT_ROOT / "test.csv",
    PROJECT_ROOT / "sample_submission.csv",
]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def has_competition_data() -> bool:
    return all(path.exists() for path in DATA_FILES)


def require_competition_data() -> None:
    if not has_competition_data():
        pytest.skip("Competition CSV files are not present in the repository clone.")
