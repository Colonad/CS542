# src/realestate/repro.py
from __future__ import annotations

import os
import random
from typing import Any, Mapping, Sequence, Dict

import numpy as np
import pandas as pd


def set_global_seed(seed: int) -> None:
    """
    Lock all major RNGs for reproducibility.

    This should be called once near the top of every training script, using
    cfg['run']['random_state'] as the seed.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Optional: if PyTorch is installed, seed it too (safe try/except)
    try:  # pragma: no cover - optional dependency
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def na_summary(
    df: pd.DataFrame,
    cols: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """
    Compact NA summary for a given dataframe and column subset.

    Returns:
      {
        "rows": int,
        "cols": int,
        "na_counts": {col: int, ...},
        "na_pct": {col: float, ...}
      }
    """
    if cols is None:
        cols = list(df.columns)
    cols = [c for c in cols if c in df.columns]

    na_counts = {c: int(df[c].isna().sum()) for c in cols}
    n_rows = max(int(len(df)), 1)
    na_pct = {c: (na_counts[c] / n_rows) * 100.0 for c in cols}

    return {
        "rows": n_rows,
        "cols": len(cols),
        "na_counts": na_counts,
        "na_pct": na_pct,
    }
