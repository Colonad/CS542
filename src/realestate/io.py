# src/realestate/io.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import hashlib
import json
import math

import numpy as np
import pandas as pd


# ----------------------------- #
# Public API
# ----------------------------- #

def load_data(
    csv_path: str | Path,
    *,
    cfg: Optional[Mapping[str, Any]] = None,
    interim_dir: str | Path = "data/interim",
    required_columns: Sequence[str] = ("price", "sold_date"),
    parse_dates: Sequence[str] = ("sold_date", "prev_sold_date"),
    drop_na_target: bool = True,
    target_col: str = "price",
    filters: Optional[Mapping[str, Mapping[str, Any]]] = None,
    winsorize_spec: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """
    Load the USA Real Estate CSV with robust hygiene and ALWAYS dump a schema snapshot.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    # --- Resolve config-backed defaults
    cfg_paths = _get(cfg, "paths", {})
    interim_dir = Path(_get(cfg_paths, "interim_dir", str(interim_dir)))

    cfg_data = _get(cfg, "data", {})
    required_columns = tuple(_get(cfg_data, "required_columns", list(required_columns)))
    cfg_parse_dates = tuple(_get(cfg_data, "parse_dates", list(parse_dates)))
    drop_na_target = bool(_get(cfg_data, "drop_na_target", drop_na_target))

    cfg_target = _get(cfg, "target", {})
    target_col = str(_get(cfg_target, "name", target_col))

    filters = _get(cfg_data, "filters", filters) or {}
    winsorize_spec = _get(_get(cfg, "features", {}), "winsorize", winsorize_spec) or {}

    # --- Load CSV
    df = pd.read_csv(csv_path, low_memory=False)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # --- Alias map (rename common variants to canonical names)
    alias_map = {
        "sold_date": [
            "sold_date", "date", "sale_date", "soldtime", "sold_time",
            "closing_date", "close_date", "transaction_date"
        ],
        "prev_sold_date": [
            "prev_sold_date", "previous_sold_date", "last_sold_date",
            "prior_sold_date", "sold_date_prev", "previous_sale_date"
        ],
        "zip_code": ["zip_code", "zipcode", "zip"],
        "house_size": ["house_size", "sqft", "square_feet", "living_area"],
        "acre_lot": ["acre_lot", "lot_size", "lot_acres", "acres"],
    }

    cols_set = set(df.columns)
    renames = {}
    for canonical, candidates in alias_map.items():
        if canonical in cols_set:
            continue
        for cand in candidates:
            if cand in cols_set:
                renames[cand] = canonical
                break
    if renames:
        df = df.rename(columns=renames)
        cols_set = set(df.columns)

    # --- Auto-detect a date-like column if 'sold_date' is still missing
    if "sold_date" not in df.columns:
        df, src = _autodetect_sold_date(df)
        cols_set = set(df.columns)

    # --- Parse dates (after aliasing/auto-detect)
    wanted_date_cols = set(cfg_parse_dates) | {"sold_date", "prev_sold_date"}
    for dcol in wanted_date_cols:
        if dcol in df.columns:
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce")

    # Replace ±inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)


    # --- NEW: drop rows with missing sold_date (can't do temporal split otherwise)
    if "sold_date" in df.columns:
        before = len(df)
        df = df.dropna(subset=["sold_date"])
        _debug_count_delta("drop_na_sold_date", before, df)



    # --- Validate required columns (after all renames)
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        dateish = [c for c in df.columns if "date" in c.lower()]
        hint = ""
        if "sold_date" in missing:
            if dateish:
                hint = f" (Found date-like columns: {dateish}. If one of these is the sale date, it should now be auto-mapped.)"
            else:
                hint = " (No date-like columns detected; dataset may not contain sale dates.)"
        raise ValueError(f"Missing required columns: {missing}{hint}")

    # --- Drop NA target if requested
    if drop_na_target:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in CSV.")
        before = len(df)
        df = df.dropna(subset=[target_col])
        _debug_count_delta("drop_na_target", before, df)

    # --- Apply broad filters (min, max, percentiles)
    if filters:
        df = _apply_filters(df, filters)

    # --- Winsorize selected columns
    if winsorize_spec and winsorize_spec.get("enabled", False):
        cols = [c for c in winsorize_spec.get("columns", []) if c in df.columns]
        if cols:
            lower = float(winsorize_spec.get("lower_pct", 0.0))
            upper = float(winsorize_spec.get("upper_pct", 99.5))
            df = _winsorize(df, cols, lower, upper)

    # --- ALWAYS dump schema snapshot
    interim_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = interim_dir / "schema.snapshot.json"
    _write_schema_snapshot(
        df=df,
        out_path=snapshot_path,
        csv_path=csv_path,
        target_col=target_col,
    )

    return df


# ----------------------------- #
# Helpers (internal)
# ----------------------------- #





def _autodetect_sold_date(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """
    If 'sold_date' is absent, try to discover a reasonable date-like column
    and map it to 'sold_date'. Preference order:
      1) exact alias matches tried earlier (handled before calling this)
      2) any column whose name contains 'sold' and 'date'
      3) any column whose name contains 'sale' and 'date'
      4) any column whose name contains 'date'
      5) any datetime64 column
    Among candidates, choose the one with the most non-null values.
    Returns (possibly-renamed df, original_column_used_or_None).
    """
    if "sold_date" in df.columns:
        return df, "sold_date"

    cols = list(df.columns)
    name_lower = {c: c.lower() for c in cols}

    def has(subs: list[str], col: str) -> bool:
        lc = name_lower[col]
        return all(s in lc for s in subs)

    # 1/2/3/4: name heuristics
    tiered: list[list[str]] = [
        ["sold", "date"],
        ["sale", "date"],
        ["date"],
    ]
    candidates: list[str] = []
    for subs in tiered:
        matches = [c for c in cols if has(subs, c)]
        if matches:
            candidates.extend(matches)

    # 5: any datetime64 columns
    dt_candidates = [c for c in cols if pd.api.types.is_datetime64_any_dtype(df[c])]
    candidates.extend([c for c in dt_candidates if c not in candidates])

    # If nothing, give up
    if not candidates:
        return df, None

    # Choose the candidate with the most non-null values after parsing attempt
    best_col = None
    best_non_null = -1
    for c in candidates:
        s = df[c]
        if not pd.api.types.is_datetime64_any_dtype(s):
            s = pd.to_datetime(s, errors="coerce")
        non_null = int(s.notna().sum())
        if non_null > best_non_null:
            best_non_null = non_null
            best_col = c

    if best_col is None:
        return df, None

    out = df.copy()
    # parse then rename to sold_date
    out[best_col] = pd.to_datetime(out[best_col], errors="coerce")
    out = out.rename(columns={best_col: "sold_date"})
    return out, best_col




def _get(dct: Optional[Mapping[str, Any]], key: str, default: Any) -> Any:
    if isinstance(dct, Mapping) and key in dct:
        return dct[key]
    return default


def _apply_filters(
    df: pd.DataFrame,
    filters: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """
    Apply per-column filters:
      - "min": numeric lower bound
      - "max": numeric upper bound
      - "max_percentile": cap values above this percentile (kept, not dropped)
      - "min_percentile": cap values below this percentile (kept, not dropped)
    """
    out = df.copy()
    for col, spec in filters.items():
        if col not in out.columns:
            continue
        s = out[col]

        # Drop by min / max (strict filtering)
        if "min" in spec:
            before = len(out)
            out = out[s >= spec["min"]]
            _debug_count_delta(f"filter[{col}>=min]", before, out)
        if "max" in spec:
            s = out[col]  # refresh after previous filter
            before = len(out)
            out = out[s <= spec["max"]]
            _debug_count_delta(f"filter[{col}<=max]", before, out)

        # Winsorize-like capping by percentiles (non-destructive to row count)
        s = out[col]
        if pd.api.types.is_numeric_dtype(s):
            if "max_percentile" in spec:
                p = float(spec["max_percentile"])
                cap = np.nanpercentile(s.to_numpy(dtype=float), p)
                out[col] = np.minimum(s, cap)
            if "min_percentile" in spec:
                p = float(spec["min_percentile"])
                floor = np.nanpercentile(s.to_numpy(dtype=float), p)
                out[col] = np.maximum(out[col], floor)

    return out


def _winsorize(
    df: pd.DataFrame,
    columns: Sequence[str],
    lower_pct: float,
    upper_pct: float,
) -> pd.DataFrame:
    """
    Winsorize selected numeric columns to [lower_pct, upper_pct] percentiles.
    """
    out = df.copy()
    lower_pct = float(lower_pct)
    upper_pct = float(upper_pct)
    for col in columns:
        if col not in out.columns:
            continue
        s = out[col]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        vals = s.to_numpy(dtype=float)
        lo = np.nanpercentile(vals, lower_pct)
        hi = np.nanpercentile(vals, upper_pct)
        out[col] = np.clip(vals, lo, hi)
    return out


def _write_schema_snapshot(
    *,
    df: pd.DataFrame,
    out_path: Path,
    csv_path: Path,
    target_col: str,
) -> None:
    """
    Compute and write a detailed schema snapshot for reproducibility.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # File hash of the raw CSV for traceability
    data_hash = _sha256_file(csv_path)

    # Column summaries
    cols_meta: Dict[str, Dict[str, Any]] = {}
    for col in df.columns:
        s = df[col]
        meta: Dict[str, Any] = {
            "dtype": str(s.dtype),
            "non_null": int(s.notna().sum()),
            "nulls": int(s.isna().sum()),
            "null_pct": float(s.isna().mean() * 100.0),
        }

        if pd.api.types.is_numeric_dtype(s):
            # Numeric summary
            arr = s.to_numpy(dtype=float)
            meta.update(
                {
                    "min": _nanfloat(np.nanmin(arr)),
                    "p1": _nanfloat(np.nanpercentile(arr, 1)),
                    "p5": _nanfloat(np.nanpercentile(arr, 5)),
                    "p50": _nanfloat(np.nanpercentile(arr, 50)),
                    "p95": _nanfloat(np.nanpercentile(arr, 95)),
                    "p99": _nanfloat(np.nanpercentile(arr, 99)),
                    "max": _nanfloat(np.nanmax(arr)),
                    "mean": _nanfloat(np.nanmean(arr)),
                    "std": _nanfloat(np.nanstd(arr, ddof=1)) if np.isfinite(np.nanstd(arr, ddof=1)) else None,
                }
            )
        elif pd.api.types.is_datetime64_any_dtype(s) or s.dtype == "object":
            # Date-ish summary (quiet coercion, then min/max)
            s_valid = _coerce_datetime_quiet(s)
            meta.update(
                {
                    "min": _ts_or_none(s_valid.min()),
                    "max": _ts_or_none(s_valid.max()),
                }
            )




        else:
            # Categorical/object: top 10 frequent values
            vc = s.astype("object").value_counts(dropna=True).head(10)
            meta["top_values"] = [
                {"value": str(idx), "count": int(cnt)} for idx, cnt in vc.items()
            ]

        cols_meta[col] = meta

    snapshot = {
        "source_csv": str(csv_path),
        "data_hash_sha256": data_hash,
        "num_rows": int(len(df)),
        "num_columns": int(df.shape[1]),
        "target_column": target_col,
        "columns": cols_meta,
    }

    out_path.write_text(json.dumps(snapshot, indent=2, default=_json_default))


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return f"sha256:{h.hexdigest()}"


def _nanfloat(x: float) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return None
        return float(x)
    except Exception:
        return None


def _ts_or_none(ts: pd.Timestamp | None) -> Optional[str]:
    if ts is None or pd.isna(ts):
        return None
    # ISO 8601
    return ts.isoformat()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return str(obj)


def _debug_count_delta(step: str, before_len: int, after_df: pd.DataFrame) -> None:
    # Quiet helper; hook up to logging if you we add logging later.
    _ = step, before_len, after_df  # no-op (placeholder for future logging)



def _coerce_datetime_quiet(s: pd.Series) -> pd.Series:
    """Coerce to datetime without 'Could not infer format...' warnings."""
    try:
        # pandas >= 2.0
        return pd.to_datetime(s, errors="coerce", format="mixed")
    except TypeError:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Could not infer format, so each element will be parsed individually",
                category=UserWarning,
                module="pandas",
            )
            return pd.to_datetime(s, errors="coerce")
