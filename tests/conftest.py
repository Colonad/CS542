# tests/conftest.py
from __future__ import annotations

import sys
from pathlib import Path

# Repo root = parent of the tests/ directory
REPO_ROOT = Path(__file__).resolve().parents[1]

# Ensure the repo root is on sys.path so "import src. ..." works
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
