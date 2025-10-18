# src/realestate/baselines.py
from __future__ import annotations

from typing import List, Sequence

import pandas as pd

__all__ = ["median_by_zip_year"]


def median_by_zip_year(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str = "price",
) -> pd.Series:
    """
    Baseline predictor: median(target) by (zip_code, year), fit on TRAIN only.

    - Groups TRAIN by ["zip_code", "year"] (when both exist).
    - Joins the resulting medians into TEST by the same keys.
    - Fills any unseen groups with the global TRAIN median(target).
    - Returns a Series aligned with TEST's index.

    Notes
    -----
    * This function is leakage-safe: it never looks at TEST targets.
    * If either "zip_code" or "year" is missing, it gracefully falls back:
        - If only "year" exists -> group by ["year"].
        - If neither exists -> use global train median for all rows.

    Parameters
    ----------
    train : pd.DataFrame
        Training split (must contain the target column).
    test : pd.DataFrame
        Test split (keys are read from here for the join).
    target : str, default "price"
        Target column to compute medians on.

    Returns
    -------
    pd.Series
        Baseline predictions (float), index-aligned to `test`.
    """
    if target not in train.columns:
        raise KeyError(f"Target column '{target}' not found in train dataframe.")

    # Ensure 'year' exists if we can derive it from 'sold_date'
    # (Upstream features.basic_time_feats should have done this already,
    #  but this makes the baseline resilient.)
    tr = train.copy()
    te = test.copy()

    if "year" not in tr.columns and "sold_date" in tr.columns and pd.api.types.is_datetime64_any_dtype(tr["sold_date"]):
        tr["year"] = tr["sold_date"].dt.year
    if "year" not in te.columns and "sold_date" in te.columns and pd.api.types.is_datetime64_any_dtype(te["sold_date"]):
        te["year"] = te["sold_date"].dt.year

    # Decide grouping columns by availability
    group_cols: List[str] = []
    if "zip_code" in tr.columns and "zip_code" in te.columns:
        group_cols.append("zip_code")
    if "year" in tr.columns and "year" in te.columns:
        group_cols.append("year")

    # Global fallback
    global_median = float(tr[target].median(skipna=True))

    # If we don't have any usable group columns, everyone gets the global median.
    if not group_cols:
        return pd.Series(global_median, index=te.index, dtype="float64")

    # Compute train medians by group (keep NaN groups separate with dropna=False).
    med_map = (
        tr.groupby(group_cols, dropna=False)[target]
        .median()
        .rename("baseline_median")
    )

    # Merge onto test using group keys; handle MultiIndex from med_map
    baseline = te.merge(
        med_map,
        how="left",
        left_on=group_cols,
        right_index=True,
        sort=False,
    )["baseline_median"].astype("float64")

    # Fill unseen groups with global train median
    baseline = baseline.fillna(global_median)

    # Ensure alignment with original test index
    baseline.index = te.index
    return baseline
