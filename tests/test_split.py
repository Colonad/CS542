# src/realestate/split.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Union
import warnings

import pandas as pd

__all__ = [
    "temporal_split",
    "temporal_3way_split",
    "split_from_cfg",
    "SplitResult",
]


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _coerce_datetime_series(s: pd.Series) -> pd.Series:
    """Coerce a series to datetime without noisy inference warnings."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    try:
        # pandas >= 2.0
        return pd.to_datetime(s, errors="raise", format="mixed")
    except TypeError:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Could not infer format, so each element will be parsed individually",
                category=UserWarning,
                module="pandas",
            )
            return pd.to_datetime(s, errors="raise")


def _assert_no_nat(df: pd.DataFrame, date_col: str) -> None:
    if df[date_col].isna().any():
        n_nat = int(df[date_col].isna().sum())
        raise ValueError(
            f"Found {n_nat} rows with NaT in '{date_col}'. "
            "Clean or drop missing dates before splitting."
        )


def _stable_sort(df: pd.DataFrame, by: str, sort_output: bool) -> pd.DataFrame:
    if sort_output:
        return df.sort_values(by, kind="stable")
    return df


def _check_min_rows(name: str, df: pd.DataFrame, min_rows: Optional[int]) -> None:
    if min_rows is not None and len(df) < int(min_rows):
        raise ValueError(f"{name} partition has {len(df)} rows, fewer than required min_rows={min_rows}.")


def _restrict_test_zip_prefixes(test: pd.DataFrame, prefixes: Iterable[str]) -> pd.DataFrame:
    if "zip_code" not in test.columns:
        return test
    prefixes = [str(p) for p in prefixes]
    if not prefixes:
        return test
    mask = pd.Series(False, index=test.index)
    z = test["zip_code"].astype(str)
    for p in prefixes:
        mask |= z.str.startswith(p)
    return test.loc[mask].copy()


def _remove_overlap(
    a: pd.DataFrame,
    b: pd.DataFrame,
    id_col: str,
    *,
    prefer: str = "a",  # keep ids in 'prefer' side; drop from the other
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove rows from the non-preferred side when the same `id_col` appears in both.
    """
    if id_col not in a.columns or id_col not in b.columns:
        return a, b

    ids_a = set(a[id_col].dropna().astype(str))
    ids_b = set(b[id_col].dropna().astype(str))
    overlap = ids_a & ids_b
    if not overlap:
        return a, b

    if prefer == "a":
        b = b[~b[id_col].astype(str).isin(overlap)].copy()
    else:
        a = a[~a[id_col].astype(str).isin(overlap)].copy()
    return a, b


# --------------------------------------------------------------------------------------
# 2-way temporal split (with optional protections)
# --------------------------------------------------------------------------------------
def temporal_split(
    df: pd.DataFrame,
    cutoff: Union[str, pd.Timestamp],
    *,
    date_col: str = "sold_date",
    sort_output: bool = True,
    # Phase-3 optional protections
    property_id_col: Optional[str] = None,
    forbid_property_cross_split: bool = False,
    restrict_test_zip_prefixes: Optional[Iterable[str]] = None,
    min_train_rows: Optional[int] = None,
    min_test_rows: Optional[int] = None,
    prefer_side: str = "train",  # which side to keep if the same property appears in both
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Deterministic temporal split with optional leakage protections.

    Train: rows with `date_col` strictly earlier than `cutoff`.
    Test : rows with `date_col` on/after `cutoff`.
    """
    if date_col not in df.columns:
        raise KeyError(f"Column '{date_col}' not found in dataframe.")

    tmp = df.copy()
    try:
        tmp[date_col] = _coerce_datetime_series(tmp[date_col])
    except Exception as e:
        raise ValueError(
            f"Column '{date_col}' is not datetime-like and could not be parsed."
        ) from e
    _assert_no_nat(tmp, date_col)

    ts = pd.Timestamp(cutoff)
    train_mask = tmp[date_col] < ts
    test_mask = ~train_mask

    train = tmp.loc[train_mask].copy()
    test = tmp.loc[test_mask].copy()

    if train.empty or test.empty:
        n_total = len(tmp)
        raise ValueError(
            "Temporal split produced an empty partition. "
            f"cutoff={ts.date()} | total={n_total} | train={len(train)} | test={len(test)}. "
            "Choose a different cutoff or verify your date distribution."
        )

    # Optional: forbid cross-property leakage
    if forbid_property_cross_split and property_id_col:
        train, test = _remove_overlap(
            train, test, property_id_col, prefer=("a" if prefer_side == "train" else "b")
        )

    # Optional: restrict test geography by ZIP prefix
    if restrict_test_zip_prefixes:
        test = _restrict_test_zip_prefixes(test, restrict_test_zip_prefixes)

    # Guardrails
    _check_min_rows("Train", train, min_train_rows)
    _check_min_rows("Test", test, min_test_rows)

    # Finalize ordering
    train = _stable_sort(train, date_col, sort_output)
    test = _stable_sort(test, date_col, sort_output)

    # Final sanity
    assert set(train.index).isdisjoint(set(test.index)), "Partitions overlap."

    protections_active = (forbid_property_cross_split and property_id_col) or bool(restrict_test_zip_prefixes)
    if protections_active:
        # When protections are on, rows can be intentionally dropped.
        assert len(train) + len(test) <= len(tmp), "Unexpected row duplication after protections."
    else:
        # Plain temporal split should cover every row exactly once.
        assert len(train) + len(test) == len(tmp), "Split does not cover all rows."

    return train, test


# --------------------------------------------------------------------------------------
# 3-way temporal split (with optional protections)
# --------------------------------------------------------------------------------------
def temporal_3way_split(
    df: pd.DataFrame,
    cutoff: Union[str, pd.Timestamp],
    val_cutoff: Union[str, pd.Timestamp],
    *,
    date_col: str = "sold_date",
    sort_output: bool = True,
    property_id_col: Optional[str] = None,
    forbid_property_cross_split: bool = False,
    restrict_test_zip_prefixes: Optional[Iterable[str]] = None,
    min_train_rows: Optional[int] = None,
    min_val_rows: Optional[int] = None,
    min_test_rows: Optional[int] = None,
    prefer_side: str = "train",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    3-way temporal split with optional leakage protections.

    Train: date < cutoff
    Val  : cutoff <= date < val_cutoff
    Test : date >= val_cutoff
    """
    if date_col not in df.columns:
        raise KeyError(f"Column '{date_col}' not found in dataframe.")

    tmp = df.copy()
    try:
        tmp[date_col] = _coerce_datetime_series(tmp[date_col])
    except Exception as e:
        raise ValueError(
            f"Column '{date_col}' is not datetime-like and could not be parsed."
        ) from e
    _assert_no_nat(tmp, date_col)

    t1 = pd.Timestamp(cutoff)
    t2 = pd.Timestamp(val_cutoff)
    if not (t1 < t2):
        raise ValueError(f"Require cutoff < val_cutoff, got {t1} !< {t2}")

    train = tmp.loc[tmp[date_col] < t1].copy()
    val = tmp.loc[(tmp[date_col] >= t1) & (tmp[date_col] < t2)].copy()
    test = tmp.loc[tmp[date_col] >= t2].copy()

    if train.empty or val.empty or test.empty:
        raise ValueError(
            "3-way temporal split produced an empty partition. "
            f"cutoff={t1.date()} | val_cutoff={t2.date()} | "
            f"train={len(train)} | val={len(val)} | test={len(test)}."
        )

    if forbid_property_cross_split and property_id_col:
        # Prefer train over val
        train, val = _remove_overlap(
            train, val, property_id_col, prefer=("a" if prefer_side == "train" else "b")
        )
        # Prefer (train ∪ val) over test
        keep_ids = set(
            pd.concat([train[property_id_col], val[property_id_col]], ignore_index=True)
            .dropna()
            .astype(str)
        )
        test = test[~test[property_id_col].astype(str).isin(keep_ids)].copy()

    if restrict_test_zip_prefixes:
        test = _restrict_test_zip_prefixes(test, restrict_test_zip_prefixes)

    # Guardrails
    _check_min_rows("Train", train, min_train_rows)
    _check_min_rows("Validation", val, min_val_rows)
    _check_min_rows("Test", test, min_test_rows)

    # Finalize ordering
    train = _stable_sort(train, date_col, sort_output)
    val = _stable_sort(val, date_col, sort_output)
    test = _stable_sort(test, date_col, sort_output)

    # Final sanity
    tr_idx, va_idx, te_idx = set(train.index), set(val.index), set(test.index)
    assert tr_idx.isdisjoint(va_idx) and tr_idx.isdisjoint(te_idx) and va_idx.isdisjoint(te_idx), "Partitions overlap."

    protections_active = (forbid_property_cross_split and property_id_col) or bool(restrict_test_zip_prefixes)
    if protections_active:
        assert len(train) + len(val) + len(test) <= len(tmp), "Unexpected row duplication after protections."
    else:
        assert len(train) + len(val) + len(test) == len(tmp), "Split does not cover all rows."

    return train, val, test


# --------------------------------------------------------------------------------------
# From-config convenience
# --------------------------------------------------------------------------------------
@dataclass
class SplitResult:
    kind: str  # "2way" or "3way"
    train: pd.DataFrame
    test: Optional[pd.DataFrame] = None
    val: Optional[pd.DataFrame] = None


def split_from_cfg(
    df: pd.DataFrame,
    cfg: Mapping[str, object],
    *,
    date_col: str = "sold_date",
) -> SplitResult:
    s = cfg.get("split", {}) or {}

    cutoff = s.get("cutoff", "2023-01-01")
    val_cutoff = s.get("val_cutoff", None)

    property_id_col = s.get("property_id_col", None)
    forbid_property_cross_split = bool(s.get("forbid_property_cross_split", False))
    restrict_test_zip_prefixes = s.get("restrict_test_zip_prefixes", None)

    min_train_rows = s.get("min_train_rows", None)
    min_val_rows = s.get("min_val_rows", None)
    min_test_rows = s.get("min_test_rows", None)

    prefer_side = s.get("prefer_side", "train")

    if val_cutoff:
        train, val, test = temporal_3way_split(
            df,
            cutoff=cutoff,
            val_cutoff=val_cutoff,
            date_col=date_col,
            property_id_col=property_id_col,
            forbid_property_cross_split=forbid_property_cross_split,
            restrict_test_zip_prefixes=restrict_test_zip_prefixes,
            min_train_rows=min_train_rows,
            min_val_rows=min_val_rows,
            min_test_rows=min_test_rows,
            prefer_side=str(prefer_side),
        )
        return SplitResult(kind="3way", train=train, val=val, test=test)

    train, test = temporal_split(
        df,
        cutoff=cutoff,
        date_col=date_col,
        property_id_col=property_id_col,
        forbid_property_cross_split=forbid_property_cross_split,
        restrict_test_zip_prefixes=restrict_test_zip_prefixes,
        min_train_rows=min_train_rows,
        min_test_rows=min_test_rows,
        prefer_side=str(prefer_side),
    )
    return SplitResult(kind="2way", train=train, test=test)
