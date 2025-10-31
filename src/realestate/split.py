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
]


# --------------------------------------------------------------------------- #
# Internal: date parsing without noisy warnings
# --------------------------------------------------------------------------- #
def _coerce_datetime_series(s: pd.Series) -> pd.Series:
    """Coerce a series to datetime without noisy inference warnings.

    Strategy:
    - If already datetime64 -> return as is.
    - Try pandas>=2.0 'format="mixed"' (avoids the 'Could not infer format' warning).
    - Fallback for older pandas: catch the specific UserWarning during parsing.
    """
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


def _stable_sort(df: pd.DataFrame, date_col: str, sort_output: bool) -> pd.DataFrame:
    return df.sort_values(date_col, kind="stable") if sort_output else df


def _as_str_series(s: pd.Series) -> pd.Series:
    # Normalize to strings without ".0", padding for typical ZIPs happens upstream in IO
    out = s.astype(str).str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    return out


def _restrict_test_zip_prefixes(test: pd.DataFrame, prefixes: Iterable[str]) -> pd.DataFrame:
    if not prefixes:
        return test
    if "zip_code" not in test.columns:
        return test
    zip_s = _as_str_series(test["zip_code"])
    prefixes = tuple(prefixes)
    mask = zip_s.str.startswith(prefixes)
    return test.loc[mask].copy()


def _remove_overlap(
    a: pd.DataFrame,
    b: pd.DataFrame,
    id_col: str,
    prefer: str = "a",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove cross-set overlaps on id_col, keeping rows on the preferred side.

    prefer: "a" or "b"
    """
    if id_col not in a.columns or id_col not in b.columns:
        return a, b
    a_ids = set(a[id_col].dropna().astype(str))
    b_ids = set(b[id_col].dropna().astype(str))
    overlap = a_ids & b_ids
    if not overlap:
        return a, b
    if prefer == "a":
        # drop from b
        b = b[~b[id_col].astype(str).isin(overlap)].copy()
    else:
        # drop from a
        a = a[~a[id_col].astype(str).isin(overlap)].copy()
    return a, b


def _check_min_rows(
    name: str, df: pd.DataFrame, minimum: Optional[int]
) -> None:
    if minimum is None:
        return
    if len(df) < int(minimum):
        raise ValueError(
            f"{name} has too few rows after split: {len(df)} < {int(minimum)}. "
            "Adjust your cutoffs, filters, or guardrails."
        )


# --------------------------------------------------------------------------- #
# Public: 2-way temporal split (train / test)
# --------------------------------------------------------------------------- #
def temporal_split(
    df: pd.DataFrame,
    cutoff: Union[str, pd.Timestamp],
    *,
    date_col: str = "sold_date",
    sort_output: bool = True,
    # Phase-3 optional protections (kept off by default for backward compat)
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

    Optional protections:
      - forbid_property_cross_split: if True and `property_id_col` is present in both
        splits, remove duplicates from the non-preferred side.
      - restrict_test_zip_prefixes: keep only test rows whose zip code starts with a
        prefix in the provided list (useful to pin evaluation geography).
      - min_train_rows / min_test_rows: guardrail row-count assertions.

    This function remains backward compatible with earlier Phases when called with
    only (df, cutoff, date_col).
    """
    if date_col not in df.columns:
        raise KeyError(f"Column '{date_col}' not found in dataframe.")

    tmp = df.copy()
    # ensure datetime dtype (strict)
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
        train, test = _remove_overlap(train, test, property_id_col, prefer="a" if prefer_side == "train" else "b")

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
    assert set(train.index).isdisjoint(set(test.index)), "Train/Test indices overlap."
    assert len(train) + len(test) == len(tmp), "Split does not cover all rows."

    return train, test


# --------------------------------------------------------------------------- #
# Public: 3-way temporal split (train / val / test)
# --------------------------------------------------------------------------- #
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
    prefer_side: str = "train",  # which side to prefer when removing overlaps
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    3-way temporal split with optional leakage protections.

    Train: rows with date < cutoff
    Val  : rows with cutoff <= date < val_cutoff
    Test : rows with date >= val_cutoff
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

    # Optional: forbid cross-property leakage across ALL pairs
    if forbid_property_cross_split and property_id_col:
        # Prefer train over val
        train, val = _remove_overlap(train, val, property_id_col, prefer="a" if prefer_side == "train" else "b")
        # Prefer (train ∪ val) over test
        if property_id_col in train.columns and property_id_col in test.columns and property_id_col in val.columns:
            # Merge train+val IDs
            keep_ids = set(pd.concat([train[property_id_col], val[property_id_col]], ignore_index=True)
                           .dropna().astype(str))
            # remove overlaps from test
            test = test[~test[property_id_col].astype(str).isin(keep_ids)].copy()

    # Optional: restrict test geography by ZIP prefix
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
    assert set(train.index).isdisjoint(set(val.index)) and set(train.index).isdisjoint(set(test.index)) \
           and set(val.index).isdisjoint(set(test.index)), "Partitions overlap."
    assert len(train) + len(val) + len(test) == len(tmp), "Split does not cover all rows."

    return train, val, test


# --------------------------------------------------------------------------- #
# Config-driven splitter (Phase 3)
# --------------------------------------------------------------------------- #
@dataclass
class SplitResult:
    kind: str                              # "2way" or "3way"
    train: pd.DataFrame
    test: pd.DataFrame
    val: Optional[pd.DataFrame] = None


def _cfg_get(cfg: dict, path: Iterable[str], default=None):
    cur = cfg
    try:
        for k in path:
            if cur is None:
                return default
            cur = cur.get(k)
        return default if cur is None else cur
    except AttributeError:
        return default


def split_from_cfg(
    df: pd.DataFrame,
    cfg: dict,
    *,
    date_col: str = "sold_date",
) -> SplitResult:
    """High-level splitter that reads all options from configs/config.yaml.

    Behavior:
      - If split.val_cutoff is null → 2-way split.
      - If split.val_cutoff is set    → 3-way split.
      - Supports property leakage blocking, test ZIP restriction, and guardrails.
    """
    split_cfg = _cfg_get(cfg, ("split",), {}) or {}

    cutoff = _cfg_get(split_cfg, ("cutoff",), None)
    if cutoff is None:
        raise KeyError("configs.split.cutoff is required.")

    val_cutoff = _cfg_get(split_cfg, ("val_cutoff",), None)
    property_id_col = _cfg_get(split_cfg, ("property_id_col",), None)
    forbid_overlap = bool(_cfg_get(split_cfg, ("forbid_property_cross_split",), False))
    restrict_zip_prefixes = _cfg_get(split_cfg, ("restrict_test_zip_prefixes",), []) or []
    min_train = _cfg_get(split_cfg, ("min_train_rows",), None)
    min_val = _cfg_get(split_cfg, ("min_val_rows",), None)
    min_test = _cfg_get(split_cfg, ("min_test_rows",), None)

    if val_cutoff:
        train, val, test = temporal_3way_split(
            df,
            cutoff=cutoff,
            val_cutoff=val_cutoff,
            date_col=date_col,
            sort_output=True,
            property_id_col=property_id_col,
            forbid_property_cross_split=forbid_overlap,
            restrict_test_zip_prefixes=restrict_zip_prefixes,
            min_train_rows=min_train,
            min_val_rows=min_val,
            min_test_rows=min_test,
            prefer_side="train",
        )
        return SplitResult(kind="3way", train=train, val=val, test=test)
    else:
        train, test = temporal_split(
            df,
            cutoff=cutoff,
            date_col=date_col,
            sort_output=True,
            property_id_col=property_id_col,
            forbid_property_cross_split=forbid_overlap,
            restrict_test_zip_prefixes=restrict_zip_prefixes,
            min_train_rows=min_train,
            min_test_rows=min_test,
            prefer_side="train",
        )
        return SplitResult(kind="2way", train=train, test=test)
