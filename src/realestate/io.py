# src/realestate/io.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Iterable

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
    Aggressive aliasing & auto-detection for sold_date and price.
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

    # --- Normalize location/category columns to strings; fix ZIP codes
    for col in ("city", "state", "zip_code", "zipcode", "zip"):
        if col in df.columns:
            if col in ("zip_code", "zipcode", "zip"):
                z = df[col]
                if pd.api.types.is_numeric_dtype(z):
                    z = pd.to_numeric(z, errors="coerce").astype("Int64").astype(str)
                else:
                    z = z.astype(str)
                z = (
                    z.str.strip()
                     .str.replace("<na>", "", regex=False)
                     .str.replace(r"\.0$", "", regex=True)
                     .str.replace(r"[^\d]", "", regex=True)
                )
                z = z.where(~z.str.fullmatch(r"\d{1,5}"), z.str.zfill(5))
                df[col] = z
            else:
                df[col] = df[col].astype(str).str.strip()

    # --- Alias map (rename common variants to canonical names)
    alias_map = {
        # sale date family (lots of real-world variants)
        "sold_date": [
            "sold_date", "date", "sale_date", "soldtime", "sold_time",
            "closing_date", "close_date", "transaction_date",
            "date_sold", "datesold", "sold_on", "soldon",
            "sale_closed_date", "sold", "sold_at",
            # feeds sometimes give generic dates; we’ll still parse & pick best later
            "record_date", "recorded_date", "list_date", "listing_date", "listed_date",
            "created_at", "updated_at", "last_update", "last_update_date",
            "timestamp", "scrape_date", "dateadded"
        ],
        # previous sale (optional)
        "prev_sold_date": [
            "prev_sold_date", "previous_sold_date", "last_sold_date",
            "prior_sold_date", "sold_date_prev", "previous_sale_date"
        ],
        # common field name variants
        "zip_code": ["zip_code", "zipcode", "zip", "postal_code", "zip code"],
        "house_size": ["house_size", "sqft", "square_feet", "living_area", "area", "living_sqft"],
        "acre_lot": ["acre_lot", "lot_size", "lot_acres", "acres", "lot_acre", "acres_lot"],
        # price family (lots of oddities)
        "price": [
            "price", "sale_price", "sold_price", "soldprice",
            "closing_price", "last_sold_price", "price_usd",
            "amount", "amount_usd", "final_price",
            "list_price", "listing_price", "median_price",
            "property_price", "home_price", "house_price",
            "price_$", "price ($)", "price_us$", "sold_usd", "price_in_usd",
            "target", "y",  # some ML-ready dumps
        ],
    }

    cols_set = set(df.columns)
    renames: Dict[str, str] = {}
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

    # --- SOLD DATE: auto-detect if still missing, else synthesize from Y/M/D parts
    if "sold_date" not in df.columns:
        df, _ = _autodetect_sold_date(df)
        cols_set = set(df.columns)
    if "sold_date" not in df.columns:
        df, _ = _synthesize_sold_date_from_parts(df)
        cols_set = set(df.columns)

    # --- Parse dates (after aliasing/auto-detect) — quiet, consistent
    wanted_date_cols = set(cfg_parse_dates) | {"sold_date", "prev_sold_date"}
    for dcol in wanted_date_cols:
        if dcol in df.columns:
            df[dcol] = _coerce_datetime_quiet(df[dcol])

    # ±inf → NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # --- PRICE: auto-detect if missing; clean to numeric always
    if "price" not in df.columns:
        df, _ = _autodetect_price(df)
        cols_set = set(df.columns)
    if "price" in df.columns:
        df["price"] = _coerce_price_to_numeric(df["price"])

    # --- Drop rows with missing sold_date only if that column exists
    #     (Some datasets truly have no sale dates; we let later split logic decide.)
    if "sold_date" in df.columns:
        before = len(df)
        df = df.dropna(subset=["sold_date"])
        _debug_count_delta("drop_na_sold_date", before, df)

    # --- Validate required columns AFTER all renames/detections
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        dateish = [c for c in df.columns if any(t in c for t in ("date", "time", "timestamp"))]
        priceish = [c for c in df.columns if any(t in c for t in ("price", "amount", "value", "cost", "usd"))]
        hint = []
        if "sold_date" in missing:
            hint.append(f"date-like columns seen: {dateish or 'none'}")
        if "price" in missing:
            hint.append(f"price-like columns seen: {priceish or 'none'}")
        txt = f" ({'; '.join(hint)})" if hint else ""
        raise ValueError(f"Missing required columns: {missing}{txt}")

    # --- Drop NA target if requested
    if drop_na_target:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in CSV.")
        before = len(df)
        df = df.dropna(subset=[target_col])
        _debug_count_delta("drop_na_target", before, df)

    # --- Filters / winsorize
    if filters:
        df = _apply_filters(df, filters)
    if winsorize_spec and winsorize_spec.get("enabled", False):
        cols = [c for c in winsorize_spec.get("columns", []) if c in df.columns]
        if cols:
            lower = float(winsorize_spec.get("lower_pct", 0.0))
            upper = float(winsorize_spec.get("upper_pct", 99.5))
            df = _winsorize(df, cols, lower, upper)

    # --- ALWAYS dump schema snapshot
    interim_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = interim_dir / "schema.snapshot.json"

    # EARLY GUARD: if all rows were dropped by filters / missing target, fail with a clear message
    if len(df) == 0:
        raise ValueError(
            "No rows remain after loading and cleaning. "
            "Likely causes: (1) 'data.filters' are too strict (e.g., price min/max), "
            "(2) 'drop_na_target' removed all rows because 'price' is missing, or "
            "(3) upstream parsing coerced your target to NaN. "
            "Relax filters in configs/config.yaml (section 'data.filters') or set 'drop_na_target: false' to inspect."
        )

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

def _get(dct: Optional[Mapping[str, Any]], key: str, default: Any) -> Any:
    if isinstance(dct, Mapping) and key in dct:
        return dct[key]
    return default


def _autodetect_sold_date(df: pd.DataFrame) -> tuple[pd.DataFrame, Optional[str]]:
    """
    If 'sold_date' is absent, discover a reasonable date-like column and map it.
    Strategy:
      A) Name heuristics (contains 'sold'+'date' / 'sale'+'date' / 'date' / 'time' / 'sold_on' / 'date_sold')
      B) Parse-ability scan over object-like columns (pick the one with the most successful parses)
      C) Already-datetime columns
    """
    if "sold_date" in df.columns:
        return df, "sold_date"

    cols = list(df.columns)
    lower = {c: c.lower() for c in cols}

    def has(subs: list[str], col: str) -> bool:
        lc = lower[col]
        return all(s in lc for s in subs)

    # A) Name-based candidates
    name_candidates: list[str] = []
    tiers: list[list[str]] = [
        ["sold", "date"],
        ["sale", "date"],
        ["date", "sold"],    # date_sold
        ["sold", "on"],      # sold_on, soldon
        ["date"],
        ["time"],
        ["timestamp"],
    ]
    for subs in tiers:
        for c in cols:
            if has(subs, c) and c not in name_candidates:
                name_candidates.append(c)

    # C) Datetime-typed
    dt_candidates = [c for c in cols if pd.api.types.is_datetime64_any_dtype(df[c])]
    for c in dt_candidates:
        if c not in name_candidates:
            name_candidates.append(c)

    candidates = name_candidates[:]

    # B) Parse-ability scan over object/string columns (broad; we’ll pick best)
    for c in cols:
        if c in candidates:
            continue
        s = df[c]
        if s.dtype == "object" or pd.api.types.is_string_dtype(s):
            parsed = _coerce_datetime_quiet(s)
            if parsed.notna().sum() >= max(10, int(0.05 * len(parsed))):  # ≥5% or ≥10 rows
                candidates.append(c)

    if not candidates:
        return df, None

    best_col, best_non_null = None, -1
    for c in candidates:
        s = df[c]
        if not pd.api.types.is_datetime64_any_dtype(s):
            s = _coerce_datetime_quiet(s)
        non_null = int(s.notna().sum())
        if non_null > best_non_null:
            best_non_null = non_null
            best_col = c

    if best_col is None:
        return df, None

    out = df.copy()
    out[best_col] = _coerce_datetime_quiet(out[best_col])
    out = out.rename(columns={best_col: "sold_date"})
    return out, best_col


def _synthesize_sold_date_from_parts(df: pd.DataFrame) -> tuple[pd.DataFrame, Optional[str]]:
    """
    Build 'sold_date' from year/month(/day) columns when no single date exists.
    Accepts many variants: sold_year, sale_year, year, yr_sold, year_sold; sold_month,
    sale_month, month, mo, mo_sold, month_sold; sold_day, sale_day, day, dom, dy.
    """
    cols = {c.lower(): c for c in df.columns}

    def _pick(candidates: list[str]) -> Optional[str]:
        for k, orig in cols.items():
            for tag in candidates:
                if tag in k:
                    return orig
        return None

    y_col = _pick(["sold_year", "sale_year", "year", "yr_sold", "year_sold"])
    m_col = _pick(["sold_month", "sale_month", "month", "mo", "mo_sold", "month_sold", "month_num"])
    d_col = _pick(["sold_day", "sale_day", "day", "dom", "dy", "day_sold", "day_num"])

    if not (y_col and m_col):
        return df, None

    out = df.copy()

    y = _coerce_int_quiet(out[y_col])
    m_raw = out[m_col]

    if pd.api.types.is_numeric_dtype(m_raw):
        m = _coerce_int_quiet(m_raw)
    else:
        m = m_raw.astype(str).str.strip().map(_month_name_to_num, na_action="ignore")
        m = pd.to_numeric(m, errors="coerce").astype("Int64")

    d = _coerce_int_quiet(out[d_col]) if d_col and d_col in out.columns else pd.Series(1, index=out.index, dtype="Int64")

    y_str = y.astype("Int64").astype(str)
    m_str = m.astype("Int64").astype(str)
    d_str = d.astype("Int64").astype(str)

    sold_date_str = y_str.str.zfill(4) + "-" + m_str.str.zfill(2) + "-" + d_str.str.zfill(2)
    out["sold_date"] = pd.to_datetime(sold_date_str, errors="coerce")

    if out["sold_date"].notna().sum() == 0:
        out.drop(columns=["sold_date"], inplace=True)
        return df, None

    return out, f"{y_col}+{m_col}" + (f"+{d_col}" if d_col else "")


def _autodetect_price(df: pd.DataFrame) -> tuple[pd.DataFrame, Optional[str]]:
    """
    If 'price' is absent, try to find a numeric/convertible column that looks like price.
    Preference order:
      1) name contains 'price'
      2) name contains one of {'amount','value','cost','usd','$'}
      3) fallback numeric heuristic: choose a numeric column in [1e3, 1e7] typical range,
         maximizing non-null count and ignoring obvious non-targets (id, bed, bath, sqft, etc.)
    """
    if "price" in df.columns:
        return df, "price"

    cols = list(df.columns)
    lower = {c: c.lower() for c in cols}

    tier1 = [c for c in cols if "price" in lower[c]]
    tier2 = [c for c in cols if any(k in lower[c] for k in ("amount", "value", "cost", "usd", "$"))]
    candidates = tier1 + [c for c in tier2 if c not in tier1]

    # Helper to score a numeric series as a plausible price
    def score_price(s: pd.Series) -> tuple[int, float]:
        s_num = _coerce_price_to_numeric(s)
        n_non_null = int(s_num.notna().sum())
        # range score: prefer typical USD home price range
        if n_non_null == 0:
            return (0, 0.0)
        try:
            med = float(np.nanmedian(s_num))
        except Exception:
            med = 0.0
        in_range = 1.0 if (1_000.0 <= med <= 10_000_000.0) else 0.2
        return (n_non_null, in_range)

    if not candidates:
        # Fallback: scan all numeric-ish columns, ignore obvious non-targets
        ignore_tokens = {"id", "bed", "bath", "acre", "lot", "sqft", "size", "area", "zip", "code", "lat", "lng"}
        for c in cols:
            lc = lower[c]
            if any(tok in lc for tok in ignore_tokens):
                continue
            s = df[c]
            if pd.api.types.is_numeric_dtype(s) or s.dtype == "object" or pd.api.types.is_string_dtype(s):
                # keep; we'll score it below
                candidates.append(c)

    if not candidates:
        return df, None

    best_col = None
    best_key = (-1, 0.0)  # (n_non_null, range_score)
    for c in candidates:
        key = score_price(df[c])
        if key > best_key:
            best_key = key
            best_col = c

    if best_col is None:
        return df, None

    out = df.copy()
    out["price"] = _coerce_price_to_numeric(out[best_col])
    return out, best_col


def _coerce_price_to_numeric(s: pd.Series) -> pd.Series:
    """Convert common currency-ish text to numeric."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    cleaned = (
        s.astype(str)
         .str.strip()
         .str.replace(r"[^\d\.\-]", "", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _coerce_int_quiet(s: pd.Series) -> pd.Series:
    """Coerce to integer safely; returns pandas nullable Int64."""
    if pd.api.types.is_integer_dtype(s):
        return s.astype("Int64")
    if pd.api.types.is_float_dtype(s):
        return pd.to_numeric(s, errors="coerce").round().astype("Int64")
    return pd.to_numeric(s.astype(str).str.strip(), errors="coerce").astype("Int64")


def _month_name_to_num(x: str) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip().lower()
    table = {
        "jan": 1, "january": 1,
        "feb": 2, "february": 2,
        "mar": 3, "march": 3,
        "apr": 4, "april": 4,
        "may": 5,
        "jun": 6, "june": 6,
        "jul": 7, "july": 7,
        "aug": 8, "august": 8,
        "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }
    return table.get(s, None)


def _apply_filters(
    df: pd.DataFrame,
    filters: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """Apply per-column min/max filters and percentile caps."""
    out = df.copy()
    for col, spec in filters.items():
        if col not in out.columns:
            continue
        s = out[col]
        if "min" in spec:
            before = len(out)
            out = out[s >= spec["min"]]
            _debug_count_delta(f"filter[{col}>=min]", before, out)
        if "max" in spec:
            s = out[col]
            before = len(out)
            out = out[s <= spec["max"]]
            _debug_count_delta(f"filter[{col}<=max]", before, out)
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
    """Winsorize selected numeric columns to [lower_pct, upper_pct] percentiles."""
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
    """Compute and write a detailed schema snapshot for reproducibility."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data_hash = _sha256_file(csv_path)

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
            # Keep only finite values; drop NaN/±inf so stats are meaningful
            arr = s.to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)]

            if arr.size == 0:
                meta.update(
                    {
                        "min": None,
                        "p1": None,
                        "p5": None,
                        "p50": None,
                        "p95": None,
                        "p99": None,
                        "max": None,
                        "mean": None,
                        "std": None,
                    }
                )
            else:
                # std with ddof=1 returns NaN for size==1; normalize to None
                _std = np.nanstd(arr, ddof=1) if arr.size > 1 else np.nan
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
                        "std": _nanfloat(_std) if np.isfinite(_std) else None,
                    }
                )

        elif pd.api.types.is_datetime64_any_dtype(s) or s.dtype == "object":
            s_valid = _coerce_datetime_quiet(s)
            meta.update({"min": _ts_or_none(s_valid.min()), "max": _ts_or_none(s_valid.max())})
        else:
            vc = s.astype("object").value_counts(dropna=True).head(10)
            meta["top_values"] = [{"value": str(idx), "count": int(cnt)} for idx, cnt in vc.items()]
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
    return ts.isoformat()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return str(obj)


def _debug_count_delta(step: str, before_len: int, after_df: pd.DataFrame) -> None:
    # Hook to add logging later if desired
    _ = step, before_len, after_df


def _coerce_datetime_quiet(s: pd.Series) -> pd.Series:
    """Coerce to datetime without 'Could not infer format...' warnings."""
    try:
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
