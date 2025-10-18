# src/realestate/split.py
from __future__ import annotations

from typing import Tuple
import warnings
import pandas as pd

__all__ = ["temporal_split"]


def _coerce_datetime_series(s: pd.Series) -> pd.Series:
    """Coerce a series to datetime without noisy inference warnings.

    Strategy:
    - If already datetime64 -> return as is.
    - Try pandas>=2.0 'format=\"mixed\"' (avoids the 'Could not infer format' warning).
    - Fallback for older pandas: catch the specific UserWarning during parsing.
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        return s

    try:
        # pandas >= 2.0 supports format="mixed" (preferred)
        return pd.to_datetime(s, errors="raise", format="mixed")
    except TypeError:
        # Older pandas: suppress only the specific inference warning
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Could not infer format, so each element will be parsed individually",
                category=UserWarning,
                module="pandas",
            )
            return pd.to_datetime(s, errors="raise")


def temporal_split(
    df: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    *,
    date_col: str = "sold_date",
    sort_output: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Deterministic temporal split with a no-leakage contract.

    Train: rows with `date_col` strictly earlier than `cutoff`.
    Test : rows with `date_col` on/after `cutoff`.
    """
    if date_col not in df.columns:
        raise KeyError(f"Column '{date_col}' not found in dataframe.")

    # Ensure datetime dtype (strict; raise on bad values)
    try:
        s = _coerce_datetime_series(df[date_col])
    except Exception as e:
        raise ValueError(
            f"Column '{date_col}' is not datetime-like and could not be parsed."
        ) from e

    tmp = df.copy()
    tmp[date_col] = s

    ts = pd.Timestamp(cutoff)

    train_mask = tmp[date_col] < ts
    test_mask = ~train_mask  # equivalent to >= ts

    # refuse NaT to keep behavior explicit
    if tmp[date_col].isna().any():
        n_nat = int(tmp[date_col].isna().sum())
        raise ValueError(
            f"Found {n_nat} rows with NaT in '{date_col}'. "
            "Clean or drop missing dates before splitting."
        )

    train = tmp.loc[train_mask].copy()
    test = tmp.loc[test_mask].copy()

    if train.empty or test.empty:
        n_total = len(tmp)
        n_train = len(train)
        n_test = len(test)
        raise ValueError(
            "Temporal split produced an empty partition. "
            f"cutoff={ts.date()} | total={n_total} | train={n_train} | test={n_test}. "
            "Choose a different cutoff or verify your date distribution."
        )

    if sort_output:
        train = train.sort_values(date_col, kind="stable")
        test = test.sort_values(date_col, kind="stable")

    # Final sanity
    assert set(train.index).isdisjoint(set(test.index)), "Train/Test indices overlap."
    assert len(train) + len(test) == len(tmp), "Split does not cover all rows."

    return train, test
