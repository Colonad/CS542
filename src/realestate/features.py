# src/realestate/features.py
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

__all__ = [
    "basic_time_feats",
    "years_since_prev",
    "apply_engineered",
    "add_zip_year_medians_train_only",
    "maybe_add_neighbors_via_cfg",
]


# --------------------------------------------------------------------------- #
# Phase 2 — Row-wise, leakage-safe time features
# --------------------------------------------------------------------------- #
def basic_time_feats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add row-wise, leakage-safe time features:
      • year  = sold_date.year
      • month = sold_date.month
      • years_since_prev_sale = (sold_date - prev_sold_date) / 365.25

    Assumptions
    -----------
    - 'sold_date' is present and already dtype datetime64 (IO layer parses).
    - No group/rolling aggregates here (purely row-wise; leakage-safe).

    Returns
    -------
    pd.DataFrame
        Copy with new columns: ['year', 'month', 'years_since_prev_sale']
    """
    if "sold_date" not in df.columns:
        raise KeyError("Column 'sold_date' is required to derive time features.")

    out = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(out["sold_date"]):
        raise TypeError(
            "'sold_date' must be datetime64 dtype before calling basic_time_feats()."
        )

    out["year"] = out["sold_date"].dt.year.astype("Int64")   # nullable integer
    out["month"] = out["sold_date"].dt.month.astype("Int64")

    # Also compute time since previous sale (row-wise; NaN if prev missing)
    out = years_since_prev(out)

    return out


def years_since_prev(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'years_since_prev_sale' = (sold_date - prev_sold_date)/365.25.

    Behavior
    --------
    - If 'prev_sold_date' is missing: create the column and fill with NaN.
    - If present but NaT: result for that row is NaN.

    Returns
    -------
    pd.DataFrame
    """
    if "sold_date" not in df.columns:
        raise KeyError("Column 'sold_date' is required to compute time deltas.")

    out = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(out["sold_date"]):
        raise TypeError("'sold_date' must be datetime64 dtype.")

    if "prev_sold_date" in out.columns:
        s_sold = out["sold_date"]
        s_prev = out["prev_sold_date"]
        # s_prev may be object or string in odd datasets; quietly coerce
        if not pd.api.types.is_datetime64_any_dtype(s_prev):
            s_prev = pd.to_datetime(s_prev, errors="coerce")
        delta_days = (s_sold - s_prev).dt.days.astype("float64")
        out["years_since_prev_sale"] = delta_days / 365.25
    else:
        out["years_since_prev_sale"] = np.nan

    return out


# --------------------------------------------------------------------------- #
# Phase 2 — Row-wise engineered features via YAML config (safe eval)
# --------------------------------------------------------------------------- #
def apply_engineered(df: pd.DataFrame, cfg: Optional[Mapping[str, Any]]) -> pd.DataFrame:
    """
    Apply purely row-wise engineered features defined in config.

    YAML example
    ------------
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
    - Only row-wise expressions are supported (no groupby/window ops here).
    - Uses a restricted eval namespace: columns + NumPy as `np`.
    - If an expression references a missing column, a clear error is raised.
    - Results are aligned to the input index.

    Returns
    -------
    pd.DataFrame
    """
    out = df.copy()
    if cfg is None:
        return out

    feats_cfg = _get(cfg, ("features", "engineered"), default=None)
    if not feats_cfg:
        return out

    # Restricted environment: no builtins; allow NumPy and columns
    safe_globals: Dict[str, Any] = {"__builtins__": {}}
    safe_locals: Dict[str, Any] = {"np": np}
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
            result = eval(expr, safe_globals, safe_locals)
        except NameError as e:
            raise NameError(
                f"Engineered feature '{name}' references unknown symbol/column: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to compute engineered feature '{name}' from expr='{expr}': {e}"
            ) from e

        # Normalize to a Series aligned to index
        if isinstance(result, (pd.Series, pd.Index)):
            ser = pd.Series(result, index=out.index)
        elif np.isscalar(result):
            ser = pd.Series(np.repeat(result, len(out)), index=out.index)
        else:
            arr = np.asarray(result)
            if arr.shape[0] != len(out):
                raise ValueError(
                    f"Engineered feature '{name}' produced length {arr.shape[0]} "
                    f"but expected {len(out)}."
                )
            ser = pd.Series(arr, index=out.index)

        out[name] = ser
        # Expose it to subsequent engineered expressions
        safe_locals[name] = out[name]

    return out


# --------------------------------------------------------------------------- #
# Phase 2 — Leakage-safe neighborhood statistics (train-only fit)
# --------------------------------------------------------------------------- #
def add_zip_year_medians_train_only(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target: str = "price",
    group_keys: list[str] | tuple[str, ...] = ("zip_code", "year"),
    out_name: str = "zip_year_price_median",
    min_group_size: int = 50,
    fill_strategy: str = "global_median",  # 'global_median' | 'zip_median'
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit group medians on TRAIN ONLY and attach leakage-safe features.

    Train feature
    -------------
    - For each row at year 'Y', attach the median(target) from the *previous* year (Y-1)
      for the same (zip_code, year) group if available; else fallback.

    Test feature
    ------------
    - For each (zip_code, year), attach the last-known training median for that ZIP
      up to that year (forward-filled over years). If missing, fallback.

    Fallbacks
    ---------
    - 'global_median' : use overall median(target) on TRAIN
    - 'zip_median'    : use per-zip median(target) on TRAIN

    Parameters
    ----------
    min_group_size : int
        Minimum rows in a (zip, year) group to consider its median reliable.

    Returns
    -------
    (train_with_feat, test_with_feat)
    """
    # Validate required columns in TRAIN
    for col in set(group_keys) | {target}:
        if col not in train_df.columns:
            raise KeyError(f"'{col}' missing from train_df; cannot build {out_name}.")
    # YEAR must be present in test too for mapping
    for col in group_keys:
        if col not in test_df.columns:
            raise KeyError(f"'{col}' missing from test_df; cannot build {out_name}.")

    # Global + per-zip fallback medians (TRAIN)
    global_median = float(train_df[target].median())
    zip_median_map = train_df.groupby("zip_code")[target].median() if "zip_code" in train_df.columns else None

    # Raw medians per (zip, year) with min_group_size filter
    grp = train_df.groupby(list(group_keys))
    med = grp[target].median().rename("med")
    cnt = grp[target].size().rename("cnt")
    gy = pd.concat([med, cnt], axis=1).reset_index()
    gy = gy[gy["cnt"] >= int(min_group_size)]

    # ---------------- TRAIN: prev-year mapping ----------------
    # Shift medians forward by +1 year so median(Y-1) maps to rows at Y.
    if "year" not in gy.columns:
        raise KeyError("'year' column is required in group_keys to build prev-year feature.")
    gy_prev = gy.copy()
    gy_prev["year"] = gy_prev["year"] + 1
    gy_prev = gy_prev.rename(columns={"med": f"{out_name}_prev"})
    train = train_df.merge(
        gy_prev[list(group_keys) + [f"{out_name}_prev"]],
        how="left",
        on=list(group_keys),
    )

    # Fallbacks for TRAIN
    if fill_strategy == "zip_median" and zip_median_map is not None and "zip_code" in train.columns:
        train[f"{out_name}_prev"] = train[f"{out_name}_prev"].fillna(
            train["zip_code"].map(zip_median_map)
        )
    else:
        train[f"{out_name}_prev"] = train[f"{out_name}_prev"].fillna(global_median)

    # ---------------- TEST: forward-filled medians ----------------
    # Forward-fill medians over years per ZIP (train-only knowledge)
    # Keep only columns needed
    keep_cols = [c for c in ["zip_code", "year", "med"] if c in gy.columns]
    zy = gy[keep_cols].copy()

    # If 'year' is not integer-typed, ensure it's numeric for sorting/ffill
    if not pd.api.types.is_integer_dtype(zy["year"]):
        zy["year"] = pd.to_numeric(zy["year"], errors="coerce").astype("Int64")

    zy = zy.sort_values(["zip_code", "year"])
    zy["med_ffill"] = zy.groupby("zip_code")["med"].ffill()
    zy_ffill = zy[["zip_code", "year", "med_ffill"]].rename(columns={"med_ffill": out_name})

    # Map to TEST on exact (zip, year); this attaches the last-known training median up to that year.
    test = test_df.merge(zy_ffill, how="left", on=["zip_code", "year"])

    # Fallbacks for TEST
    if fill_strategy == "zip_median" and zip_median_map is not None and "zip_code" in test.columns:
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
    If enabled in config under `features.leakage_safe_neighbors`, attach train-only
    neighborhood statistics (currently: median by ['zip_code','year']).

    Config example
    --------------
    features:
      leakage_safe_neighbors:
        enabled: true
        group_keys: [zip_code, year]
        stats:
          - kind: median
            of: price
            name: zip_year_price_median
        min_group_size: 50
        fill_strategy: global_median   # or 'zip_median'

    Returns
    -------
    (train_df_with_feats, test_df_with_feats, added_feature_names)
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

    tr, te = train_df.copy(), test_df.copy()

    for spec in stats:
        kind = spec.get("kind")
        if kind != "median":
            raise NotImplementedError(f"Only 'median' statistic is supported in Phase 2. Got: {kind!r}")
        of = spec.get("of", target)
        name = spec.get("name", "zip_year_price_median")

        tr, te = add_zip_year_medians_train_only(
            tr,
            te,
            target=of,
            group_keys=group_keys,
            out_name=name,
            min_group_size=min_group_size,
            fill_strategy=fill_strategy,
        )
        # Track both the test feature and the train prev-year feature name
        added.append(name)
        added.append(f"{name}_prev")

    return tr, te, added


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
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
