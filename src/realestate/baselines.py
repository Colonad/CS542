# src/realestate/baselines.py
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

__all__ = ["median_by_zip_year"]


def _ensure_year(df: pd.DataFrame, *, date_col: str = "sold_date") -> pd.DataFrame:
    """
    Ensure a nullable integer 'year' column exists.
    If `sold_date` exists and is datetime64, derive it; otherwise leave as-is.
    """
    out = df.copy()
    if "year" not in out.columns and date_col in out.columns and pd.api.types.is_datetime64_any_dtype(out[date_col]):
        out["year"] = out[date_col].dt.year.astype("Int64")
    return out


def _normalize_zip_code(s: pd.Series) -> pd.Series:
    """
    Normalize ZIP codes to strings and (when 1–5 digits) zero-pad to 5 characters.
    Keeps non-US or extended ZIP+4 forms as best-effort strings.
    """
    # If numeric-like, coerce carefully (avoid '12345.0')
    if pd.api.types.is_numeric_dtype(s):
        s = pd.to_numeric(s, errors="coerce").astype("Int64").astype(str)
    else:
        s = s.astype("string")

    # Clean and normalize
    s = (
        s.str.strip()
         .str.replace("<NA>", "", regex=False)
         .str.replace(r"\.0$", "", regex=True)
         .str.replace(r"[^\d]", "", regex=True)  # keep only digits when possible
    )
    # Zero-pad 1–5 pure-digit ZIPs
    mask = s.str.fullmatch(r"\d{1,5}")
    s = s.where(~mask, s.str.zfill(5))
    return s


def median_by_zip_year(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target: str = "price",
    date_col: str = "sold_date",
    min_group_size: int = 1,
) -> pd.Series:
    """
    Leakage-safe baseline predictor: median(target) by (zip_code, year), fit on TRAIN only.

    Mapping priority (strict backoffs):
        1) (zip_code, year) median when group size >= min_group_size
        2) zip_code-only median (size >= min_group_size)
        3) year-only median (size >= min_group_size)
        4) global TRAIN median

    The function never looks at TEST targets and returns a Series aligned to TEST.index.

    Parameters
    ----------
    train : pd.DataFrame
        Training split; must contain `target`. Uses `zip_code`, `year` (or derives from `sold_date`) if present.
    test : pd.DataFrame
        Test split to which the medians are mapped.
    target : str, default "price"
        Target column name.
    date_col : str, default "sold_date"
        Column used to derive `year` when `year` is missing.
    min_group_size : int, default 1
        Minimum number of rows required for a group to contribute a median;
        smaller groups fall back to the next level.

    Returns
    -------
    pd.Series
        Baseline predictions (float64), index-aligned to `test`.
    """
    if target not in train.columns:
        raise KeyError(f"Target column '{target}' not found in train dataframe.")

    # Work on copies; keep upstream data pristine
    tr = train.copy()
    te = test.copy()

    # Ensure year exists (derivable from sold_date), normalize ZIPs to robust strings
    tr = _ensure_year(tr, date_col=date_col)
    te = _ensure_year(te, date_col=date_col)

    if "zip_code" in tr.columns:
        tr["zip_code"] = _normalize_zip_code(tr["zip_code"])
    if "zip_code" in te.columns:
        te["zip_code"] = _normalize_zip_code(te["zip_code"])

    # Global fallback
    global_median = float(pd.to_numeric(tr[target], errors="coerce").median(skipna=True))

    # If neither key is available, return global median for everyone
    has_zip = "zip_code" in tr.columns and "zip_code" in te.columns
    has_year = "year" in tr.columns and "year" in te.columns

    if not (has_zip or has_year):
        return pd.Series(global_median, index=te.index, dtype="float64")

    # Helper: safe groupby median with a row-count filter
    def _group_median(df: pd.DataFrame, keys: List[str]) -> Optional[pd.Series]:
        # Only include keys that actually exist (caller ensures presence)
        if not all(k in df.columns for k in keys):
            return None
        g = df.groupby(keys, dropna=False)[target]
        stats = pd.DataFrame(
            {"med": pd.to_numeric(g.median(), errors="coerce"), "n": g.size()}
        )
        stats = stats[stats["n"] >= int(min_group_size)]
        if stats.empty:
            return None
        return stats["med"]

    # Compute train medians with descending specificity
    med_zy = _group_median(tr, ["zip_code", "year"]) if (has_zip and has_year) else None
    med_z = _group_median(tr, ["zip_code"]) if has_zip else None
    med_y = _group_median(tr, ["year"]) if has_year else None

    # Start with all-NaN Series aligned to test
    baseline = pd.Series(np.nan, index=te.index, dtype="float64")

    # 1) map (zip, year)
    if med_zy is not None:
        # Join on MultiIndex
        baseline = (
            te.merge(
                med_zy.rename("m1"),
                how="left",
                left_on=["zip_code", "year"],
                right_index=True,
                sort=False,
            )["m1"]
            .astype("float64")
            .reindex(te.index)
        )

    # 2) fill with zip-only medians
    if med_z is not None:
        needs = baseline.isna()
        if needs.any():
            fill_zip = (
                te.loc[needs]
                .merge(
                    med_z.rename("m2"),
                    how="left",
                    left_on=["zip_code"],
                    right_index=True,
                    sort=False,
                )["m2"]
                .astype("float64")
            )
            baseline.loc[needs] = fill_zip.values

    # 3) fill with year-only medians
    if med_y is not None:
        needs = baseline.isna()
        if needs.any():
            fill_year = (
                te.loc[needs]
                .merge(
                    med_y.rename("m3"),
                    how="left",
                    left_on=["year"],
                    right_index=True,
                    sort=False,
                )["m3"]
                .astype("float64")
            )
            baseline.loc[needs] = fill_year.values

    # 4) final global fallback
    baseline = baseline.fillna(global_median).astype("float64")
    # Ensure the same index as original test
    baseline.index = te.index
    return baseline
