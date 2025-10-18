# src/realestate/features.py
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

__all__ = [
    "basic_time_feats",
    "years_since_prev",
    "apply_engineered",
]


def basic_time_feats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add row-wise, leakage-safe time features:
      - year, month (from 'sold_date')
      - years_since_prev_sale (from 'prev_sold_date' if present; else NaN)

    Notes
    -----
    - Assumes `sold_date` is already datetime64 (IO layer handles parsing).
    - Does *not* compute any group/rolling aggregates (saved for later phases).
    """
    if "sold_date" not in df.columns:
        raise KeyError("Column 'sold_date' is required to derive time features.")

    out = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(out["sold_date"]):
        # Be strict: upstream IO should have parsed this already.
        raise TypeError("'sold_date' must be datetime64 dtype before calling basic_time_feats().")

    out["year"] = out["sold_date"].dt.year.astype("Int64")  # nullable integer
    out["month"] = out["sold_date"].dt.month.astype("Int64")

    # Include years_since_prev_sale as part of the basic set (tests expect this)
    out = years_since_prev(out)

    return out


def years_since_prev(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `years_since_prev_sale` = (sold_date - prev_sold_date)/365.25.
    If 'prev_sold_date' is missing, the column is created and filled with NaN.
    """
    out = df.copy()
    if "prev_sold_date" in out.columns:
        s_sold = out["sold_date"]
        s_prev = out["prev_sold_date"]
        if not pd.api.types.is_datetime64_any_dtype(s_sold):
            raise TypeError("'sold_date' must be datetime64 dtype.")
        # prev may be all-NaT; that's fine—result will be NaN
        # Compute in days, then convert to years
        delta_days = (s_sold - s_prev).dt.days.astype("float64")
        out["years_since_prev_sale"] = delta_days / 365.25
    else:
        out["years_since_prev_sale"] = np.nan

    return out


def apply_engineered(df: pd.DataFrame, cfg: Optional[Mapping[str, Any]]) -> pd.DataFrame:
    """
    Apply purely row-wise engineered features defined in config:

    YAML example:
    features:
      engineered:
        - name: bed_per_bath
          expr: "bed / np.clip(bath, 1, None)"
        - name: lot_per_size
          expr: "acre_lot / np.clip(house_size, 1, None)"
        - name: log_house_size
          expr: "np.log1p(house_size)"

    Rules
    -----
    - Only row-wise expressions are supported (no groupby / windowing here).
    - Uses a restricted eval namespace: columns + NumPy as `np`.
    - If an expression references a missing column, it raises a clear error.
    - Resulting series are aligned to the input index.
    """
    out = df.copy()
    if cfg is None:
        return out

    feats_cfg = _get(cfg, ("features", "engineered"), default=None)
    if not feats_cfg:
        return out

    # Prepare a restricted evaluation environment
    safe_globals: Dict[str, Any] = {"__builtins__": {}}
    safe_locals: Dict[str, Any] = {"np": np}
    # expose columns by name (Series)
    for col in out.columns:
        safe_locals[col] = out[col]

    for item in feats_cfg:
        name = item.get("name")
        expr = item.get("expr")
        if not name or not isinstance(name, str):
            raise ValueError(f"Engineered feature missing valid 'name': {item!r}")
        if not expr or not isinstance(expr, str):
            raise ValueError(f"Engineered feature '{name}' missing valid 'expr' string.")

        try:
            # Evaluate expression; allow numpy via `np` and columns as Series
            result = eval(expr, safe_globals, safe_locals)
        except NameError as e:
            raise NameError(
                f"Engineered feature '{name}' references unknown symbol/column: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to compute engineered feature '{name}' from expr='{expr}': {e}"
            ) from e

        # Normalize result to a pandas Series aligned to df.index
        if isinstance(result, (pd.Series, pd.Index)):
            ser = pd.Series(result, index=out.index)
        elif np.isscalar(result):
            ser = pd.Series(np.repeat(result, len(out)), index=out.index)
        else:
            arr = np.asarray(result)
            if arr.shape[0] != len(out):
                raise ValueError(
                    f"Engineered feature '{name}' produced length {arr.shape[0]} but expected {len(out)}."
                )
            ser = pd.Series(arr, index=out.index)

        out[name] = ser

        # Also expose the new column for potential downstream engineered features
        safe_locals[name] = out[name]

    return out


# ----------------------------- #
# Internal helpers
# ----------------------------- #

def _get(cfg: Optional[Mapping[str, Any]], path: Iterable[str], default: Any = None) -> Any:
    cur: Any = cfg
    try:
        for key in path:
            if cur is None:
                return default
            cur = cur.get(key)
        return default if cur is None else cur
    except AttributeError:
        return default
