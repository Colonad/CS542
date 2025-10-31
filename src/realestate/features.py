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



# ----------------------------- #
# Phase-2: leakage-safe neighbors (train-only fit)
# ----------------------------- #

def add_zip_year_medians_train_only(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target: str = "price",
    group_keys: list[str] | tuple[str, ...] = ("zip_code", "year"),
    out_name: str = "zip_year_price_median",
    min_group_size: int = 50,
    fill_strategy: str = "global_median",   # 'global_median' | 'zip_median'
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit group medians on TRAIN ONLY and attach leakage-safe features:

    - TRAIN: previous-year median per zip (i.e., year-1). If unavailable, fallback.
    - TEST: last-known training median for that (zip, year) ffilled over years. If unavailable, fallback.

    Returns updated (train_df, test_df).
    """
    for col in ("zip_code", "year", target):
        if col not in train_df.columns:
            raise KeyError(f"'{col}' missing from train_df; cannot build {out_name}.")

    # --- Compute global + per-zip fallbacks from TRAIN
    global_median = float(train_df[target].median())
    zip_median_map = train_df.groupby("zip_code")[target].median()

    # --- Raw medians per (zip, year) with support for min_group_size
    grp = train_df.groupby(list(group_keys))
    counts = grp[target].size().rename("cnt")
    med = grp[target].median().rename("med")
    gy = pd.concat([med, counts], axis=1).reset_index()
    gy = gy[gy["cnt"] >= int(min_group_size)]  # enforce support

    # -------------------- TRAIN FEATURE (prev-year median) --------------------
    gy_prev = gy.copy()
    gy_prev["year"] = gy_prev["year"] + 1          # shift forward: median(Y-1) will map to rows at Y
    gy_prev = gy_prev.rename(columns={"med": f"{out_name}_prev"})
    # Merge into train on (zip, year)
    train = train_df.merge(
        gy_prev[["zip_code", "year", f"{out_name}_prev"]],
        how="left",
        on=["zip_code", "year"],
    )

    # Fallbacks for TRAIN
    if fill_strategy == "zip_median":
        train[f"{out_name}_prev"] = train[f"{out_name}_prev"].fillna(
            train["zip_code"].map(zip_median_map)
        )
    else:
        train[f"{out_name}_prev"] = train[f"{out_name}_prev"].fillna(global_median)

    # -------------------- TEST FEATURE (ffilled training medians) --------------------
    # Build a dense (zip, year) index covering training years we’ve seen
    zy = gy[["zip_code", "year", "med"]].copy()
    # For each zip, sort by year and forward-fill medians across years
    zy = zy.sort_values(["zip_code", "year"])
    zy["med_ffill"] = zy.groupby("zip_code")["med"].ffill()
    zy_ffill = zy[["zip_code", "year", "med_ffill"]].rename(columns={"med_ffill": out_name})

    # Map to TEST on exact (zip, year)
    test = test_df.merge(zy_ffill, how="left", on=["zip_code", "year"])

    # Fallbacks for TEST
    if fill_strategy == "zip_median":
        test[out_name] = test[out_name].fillna(test["zip_code"].map(zip_median_map))
    else:
        test[out_name] = test[out_name].fillna(global_median)

    return train, test


def maybe_add_neighbors_via_cfg(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Optional[Mapping[str, Any]],
    *,
    target: str = "price",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Read `features.leakage_safe_neighbors` from config and attach features if enabled.

    Returns (train_df, test_df, added_feature_names)
    """
    added: list[str] = []
    if cfg is None:
        return train_df, test_df, added

    block = _get(cfg, ("features", "leakage_safe_neighbors"), default=None)
    if not block or not block.get("enabled", False):
        return train_df, test_df, added

    group_keys = block.get("group_keys", ["zip_code", "year"])
    stats = block.get("stats", [{"kind": "median", "of": target, "name": "zip_year_price_median"}])
    min_group_size = int(block.get("min_group_size", 50))
    fill_strategy = str(block.get("fill_strategy", "global_median"))

    for spec in stats:
        if spec.get("kind") != "median":
            raise NotImplementedError(f"Only 'median' supported in Phase-2. Got: {spec}")
        of = spec.get("of", target)
        name = spec.get("name", "zip_year_price_median")
        train_df, test_df = add_zip_year_medians_train_only(
            train_df,
            test_df,
            target=of,
            group_keys=group_keys,
            out_name=name,
            min_group_size=min_group_size,
            fill_strategy=fill_strategy,
        )
        added.append(name)
        # For training rows we produced "<name>_prev"; include it as well
        added.append(f"{name}_prev")

    return train_df, test_df, added
