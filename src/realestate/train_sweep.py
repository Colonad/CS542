#!/usr/bin/env python3
"""
src/realestate/train_sweep.py
=============================

Phase 6 — Model sweep (small grid)

- 3-way temporal split:
    train ≤ 2022, val = 2023, test ≥ 2024
- Model families and grids:
    * Ridge: alpha ∈ {0.5, 1, 2, 5}
    * RandomForestRegressor:
        n_estimators ∈ {400, 800}
        min_samples_leaf ∈ {1, 2, 4}
        max_depth ∈ {None, 20, 40}
    * XGBRegressor (if enabled): grid over learning_rate, max_depth,
        n_estimators, subsample, colsample_bytree (small grid, overrideable)

Artifacts written under paths.out_dir:
  - run/<run_id>/{metrics,preds,models}/...
  - metrics/phase6_sweep_metrics.json (consolidated)
  - preds/test_preds_*.csv (per final model)
  - models/*.pkl (fitted pipelines)
  - models/feature_importances_*.csv (RF/XGB grouped importances if available)
  - run/.../metrics/config.snapshot.yaml (config copy)

Requires:
  - configs/config.yaml
  - src/realestate/io.py          (load_data)
  - src/realestate/features.py    (basic_time_feats, years_since_prev,
                                   apply_engineered, maybe_add_neighbors_via_cfg)
  - src/realestate/modeling.py    (preprocessor)
  - src/realestate/baselines.py   (median_by_zip_year)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple
from datetime import datetime
import hashlib
import math
from copy import deepcopy

import numpy as np
import pandas as pd
import yaml

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import scipy.sparse as sp

from joblib import dump as joblib_dump

# Project modules
from . import io as io_mod
from .features import (
    basic_time_feats,
    years_since_prev,
    apply_engineered,
    maybe_add_neighbors_via_cfg,
)
from .baselines import median_by_zip_year
from .modeling import preprocessor, describe_estimator

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover (defensive)
    tqdm = None


# --------------------------------------------------------------------------------------
# Lightweight progress helper (same behavior as previous files)
# --------------------------------------------------------------------------------------
class Prog:
    def __init__(self, enabled: bool, total: int, desc: str):
        self.enabled = enabled and (tqdm is not None)
        self.t = tqdm(total=total, desc=desc, leave=True) if self.enabled else None

    def step(self, msg: str):
        if self.t:
            self.t.set_postfix_str(str(msg)[:60], refresh=True)
            self.t.update(1)

    def close(self):
        if self.t:
            self.t.close()


# --------------------------------------------------------------------------------------
# Optional cuML availability / helpers
# --------------------------------------------------------------------------------------
def _probe_cuml() -> dict:
    """
    Try to import cuML and return metadata:
      {
        "available": bool,
        "version": str | None,
        "ridge_cls": class | None,
        "rf_cls": class | None,
        "note": str
      }
    """
    info = {
        "available": False,
        "version": None,
        "ridge_cls": None,
        "rf_cls": None,
        "note": "cuML not imported",
    }
    try:
        import cuml  # type: ignore
        from cuml.linear_model import Ridge as cuRidge  # type: ignore
        from cuml.ensemble import RandomForestRegressor as cuRF  # type: ignore

        info["available"] = True
        info["version"] = getattr(cuml, "__version__", "unknown")
        info["ridge_cls"] = cuRidge
        info["rf_cls"] = cuRF
        info["note"] = "cuML available"
    except Exception as e:
        info["note"] = f"cuML unavailable: {e}"
    return info


class Densify(TransformerMixin, BaseEstimator):
    """Ensure dense ndarray output (float32). Needed for cuML estimators."""
    def fit(self, X, y=None):  # noqa: D401
        return self
    def transform(self, X):
        if sp.issparse(X):
            X = X.toarray()
        return np.asarray(X, dtype=np.float32)


# --------------------------------------------------------------------------------------
# XGBoost GPU probe / device introspection
# --------------------------------------------------------------------------------------
def _probe_xgb_gpu() -> tuple[bool, str, dict]:
    """
    Return (ok, note, suggested_params_for_gpu).

    We try both modern ('device'='cuda', 'tree_method'='hist') and legacy
    ('tree_method'='gpu_hist','predictor'='gpu_predictor') recipes, then
    pick whichever actually uses the GPU at runtime.
    """
    try:
        import xgboost as xgb
    except Exception as e:
        return False, f"xgboost import failed: {e}", {}

    ver = getattr(xgb, "__version__", "unknown")
    try:
        major = int(str(ver).split(".")[0])
    except Exception:
        major = 0

    # tiny synthetic train on purpose
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 16)).astype(np.float32)
    y = rng.normal(size=(64,)).astype(np.float32)
    dtrain = xgb.DMatrix(X, label=y)

    def _deep_get(obj, key):
        if isinstance(obj, dict):
            if key in obj:
                return obj[key]
            for v in obj.values():
                r = _deep_get(v, key)
                if r is not None:
                    return r
        elif isinstance(obj, list):
            for v in obj:
                r = _deep_get(v, key)
                if r is not None:
                    return r
        return None

    def _runtime_device(booster) -> str:
        try:
            cfg = json.loads(booster.save_config())
            dev = _deep_get(cfg, "device")
            if dev:
                dev = str(dev).lower()
                if dev.startswith("cuda"):
                    return "cuda"
                return dev
            tm = _deep_get(cfg, "tree_method")
            if tm and "gpu" in str(tm).lower():
                return "gpu"
            pred = _deep_get(cfg, "predictor")
            if pred and "gpu" in str(pred).lower():
                return "gpu"
            return "cpu"
        except Exception:
            return "unknown"

    notes = []
    # Try modern recipe first (xgb >= 2 prefers 'device')
    try:
        params_modern = {"device": "cuda", "tree_method": "hist", "verbosity": 0}
        booster = xgb.train(params_modern, dtrain, num_boost_round=2)
        used = _runtime_device(booster)
        if used in ("cuda", "gpu"):
            # Add predictor to keep the sklearn wrapper on GPU for predict as well
            params_modern["predictor"] = "gpu_predictor"
            return True, f"xgboost {ver} (device=cuda OK)", params_modern
        notes.append(f"modern recipe used={used}")
    except Exception as e:
        notes.append(f"modern recipe failed: {e}")

    # Fallback to legacy explicit GPU recipe
    try:
        params_legacy = {"tree_method": "gpu_hist", "predictor": "gpu_predictor", "verbosity": 0}
        booster = xgb.train(params_legacy, dtrain, num_boost_round=2)
        used = _runtime_device(booster)
        if used in ("cuda", "gpu"):
            return True, f"xgboost {ver} (gpu_hist OK)", params_legacy
        notes.append(f"legacy recipe used={used}")
    except Exception as e:
        notes.append(f"legacy recipe failed: {e}")

    return False, f"xgboost {ver} GPU probe failed; " + " | ".join(notes), {}


def _xgb_runtime_device(pipe: Pipeline) -> str:
    """Inspect fitted XGB pipeline and return one of {'cuda','gpu','cpu','unknown'}."""
    try:
        mdl = pipe.named_steps.get("mdl")
        if mdl is None or not hasattr(mdl, "get_booster"):
            return "unknown"
        cfg = json.loads(mdl.get_booster().save_config())

        def _deep_get(obj, key):
            if isinstance(obj, dict):
                if key in obj:
                    return obj[key]
                for v in obj.values():
                    r = _deep_get(v, key)
                    if r is not None:
                        return r
            elif isinstance(obj, list):
                for v in obj:
                    r = _deep_get(v, key)
                    if r is not None:
                        return r
            return None

        dev = _deep_get(cfg, "device")
        if dev:
            dev = str(dev).lower()
            if dev.startswith("cuda"):
                return "cuda"
            return dev
        tm = _deep_get(cfg, "tree_method")
        if tm and "gpu" in str(tm).lower():
            return "gpu"
        return "cpu"
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------------------------
def _load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p.resolve()}")
    data = yaml.safe_load(p.read_text())
    if data is None or not isinstance(data, dict) or not data:
        raise ValueError(
            f"Config file is empty or invalid YAML: {p.resolve()}\n"
            "Please populate configs/config.yaml."
        )
    return data


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk_size), b""):
            h.update(b)
    return f"sha256:{h.hexdigest()}"


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return MAE/RMSE/R2 as floats (portable across sklearn versions)."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def _duan_smearing_factor(y_fit_true: np.ndarray, y_fit_pred: np.ndarray) -> float:
    """
    Duan's smearing estimate for log-link inversions:
      y = log1p(price). If model predicts y_hat, then on the original scale:
      price_hat ≈ expm1(y_hat) * E[exp(ε)], where ε = y - y_hat.
    """
    eps = (y_fit_true - y_fit_pred).astype(float)
    return float(np.mean(np.exp(eps)))


def _inverse_target(pred_fit: np.ndarray, use_log: bool, smear: Optional[float]) -> np.ndarray:
    if not use_log:
        return pred_fit
    out = np.expm1(pred_fit)
    if smear is not None and np.isfinite(smear):
        out = out * smear
    return out


def _safe_model_params(pipe: Pipeline) -> dict[str, Any]:
    """
    Extract compact, JSON-safe model params (robust to None/NaN/arrays),
    flattening only the top-level estimator params.
    """
    try:
        mdl = pipe.named_steps.get("mdl")
        if mdl is None:
            return {}
        params = mdl.get_params(deep=False)
        out: dict[str, Any] = {}
        for k, v in params.items():
            try:
                if isinstance(v, (list, tuple)):
                    sv = str(v)
                    out[k] = sv if len(sv) <= 200 else f"<{type(v).__name__} len={len(v)}>"
                elif isinstance(v, dict):
                    sv = str({kk: type(vv).__name__ for kk, vv in v.items()})
                    out[k] = sv if len(sv) <= 200 else f"<dict keys={len(v)}>"
                elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    out[k] = None
                else:
                    out[k] = v
            except Exception:
                out[k] = str(type(v))
        return out
    except Exception:
        return {}


def _ensure_datetime(df: pd.DataFrame, col: str = "sold_date") -> None:
    if col not in df.columns:
        raise KeyError(f"'{col}' missing.")
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    if df[col].isna().any():
        # Keep NaT rows; they will be excluded by temporal filtering as needed
        pass


def temporal_three_way_split(
    df: pd.DataFrame,
    *,
    date_col: str = "sold_date",
    train_max_year: int = 2022,
    val_year: int = 2023,
    test_min_year: int = 2024,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _ensure_datetime(df, date_col)
    y = df[date_col].dt.year
    tr = df[y <= train_max_year].copy()
    va = df[y == val_year].copy()
    te = df[y >= test_min_year].copy()
    return tr, va, te


# --------------------------------------------------------------------------------------
# Estimator builders (sklearn vs cuML) and pipeline
# --------------------------------------------------------------------------------------
def _to_float32(X):
    try:
        return X.astype(np.float32)
    except Exception:
        if sp.issparse(X):
            return X.asfptype()
        return np.asarray(X, dtype=np.float32)


def _is_cuml_estimator(estimator) -> bool:
    mod = getattr(estimator.__class__, "__module__", "")
    return mod.startswith("cuml")


def _build_pipeline(estimator, num_cols: list[str], cat_cols: list[str], cfg: Mapping[str, Any]) -> Pipeline:
    pre = preprocessor(num_cols, cat_cols, cfg=cfg)
    cast32 = FunctionTransformer(_to_float32, accept_sparse=True, feature_names_out="one-to-one")
    steps = [("pre", pre)]

    # If using cuML, force dense float32 right after preprocessing
    if _is_cuml_estimator(estimator):
        steps.append(("densify", Densify()))
    else:
        # keep feature matrices as-is (may be sparse for sklearn Ridge/RF)
        pass

    steps.append(("cast32", cast32))
    steps.append(("mdl", estimator))
    return Pipeline(steps)


def _init_with_fallback(cls, **kwargs):
    """
    Try to construct estimator; if TypeError (unknown kwargs), drop common keys and retry.
    """
    try:
        return cls(**kwargs)
    except TypeError:
        # drop a few common sklearn-only keys that cuML may not accept
        bad_keys = ("min_samples_leaf", "n_jobs", "random_state", "nthread")
        kw2 = {k: v for k, v in kwargs.items() if k not in bad_keys}
        return cls(**kw2)


def _build_ridge(alpha: float, rs: int, use_cuml: bool, cu_info: dict):
    if use_cuml and cu_info["available"] and cu_info["ridge_cls"] is not None:
        # cuML Ridge typically accepts alpha; random_state may not be supported -> init fallback
        est = _init_with_fallback(cu_info["ridge_cls"], alpha=float(alpha), random_state=int(rs))
        return est, "gpu(cuml)"
    # sklearn fallback
    return Ridge(alpha=float(alpha), random_state=int(rs)), "cpu(sklearn)"


def _build_rf(spec: dict, rs: int, use_cuml: bool, cu_info: dict):
    """
    Build a RandomForest for either cuML (GPU) or sklearn (CPU).

    cuML notes:
      - Accepts n_estimators, max_depth (int), random_state.
      - Ignores sklearn-only knobs like min_samples_leaf, n_jobs.
      - If max_depth is None in the grid, we DO NOT pass it (let cuML default).
    """
    if use_cuml and cu_info["available"] and cu_info["rf_cls"] is not None:
        params = {
            "n_estimators": int(spec["n_estimators"]),
            "random_state": int(rs),
        }
        md = spec.get("max_depth", None)
        if md is not None:
            params["max_depth"] = int(md)  # only pass when it’s an int
        est = _init_with_fallback(cu_info["rf_cls"], **params)
        return est, "gpu(cuml)"
    # sklearn fallback (CPU)
    return RandomForestRegressor(
        n_estimators=int(spec["n_estimators"]),
        min_samples_leaf=int(spec["min_samples_leaf"]),
        max_depth=spec["max_depth"],
        n_jobs=-1,
        random_state=int(rs),
    ), "cpu(sklearn)"


def _prep_features_and_split(cfg: Mapping[str, Any], p: Prog) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    list[str], list[str], list[str]
]:
    """Load data, build features (row-wise + engineered), split, then add neighbor stats safely."""
    data_csv = Path(cfg["paths"]["data_csv"])
    df = io_mod.load_data(data_csv, cfg=cfg)  # enforces schema, parses dates
    p.step("Loaded CSV + schema snapshot")

    # Row-wise time features (safe)
    df = basic_time_feats(df)
    df = apply_engineered(df, cfg)
    p.step("Derived row-wise + engineered features")

    # 3-way split
    tr_df, va_df, te_df = temporal_three_way_split(
        df,
        date_col="sold_date",
        train_max_year=int(cfg.get("split", {}).get("train_max_year", 2022)),
        val_year=int(cfg.get("split", {}).get("val_year", 2023)),
        test_min_year=int(cfg.get("split", {}).get("test_min_year", 2024)),
    )
    p.step("Temporal split: train/val/test")

    # Feature lists from config
    feats_cfg = cfg.get("features", {})
    num_cols = [c for c in feats_cfg.get("numeric", []) if c in df.columns]
    cat_cols = [c for c in feats_cfg.get("categorical", []) if c in df.columns]
    target_col = cfg.get("target", {}).get("name", "price")
    use_log = bool(cfg.get("target", {}).get("log_transform", True))

    # Neighbor features for VAL: fit on TRAIN only, map to VAL
    tr_df_aug, va_df_aug, added_va = maybe_add_neighbors_via_cfg(tr_df, va_df, cfg, target=target_col)
    if added_va:
        p.step(f"Neighbors (train→val): {', '.join(sorted(set(added_va)))}")

    # Ensure neighbor numeric feats are included
    for f in added_va:
        if f in tr_df_aug.columns and pd.api.types.is_numeric_dtype(tr_df_aug[f]):
            if f not in num_cols:
                num_cols.append(f)

    # For TEST: fit neighbors on TRAIN+VAL only, map to TEST
    trainval_df = pd.concat([tr_df, va_df], axis=0, ignore_index=True)
    trainval_aug, te_df_aug, added_te = maybe_add_neighbors_via_cfg(trainval_df, te_df, cfg, target=target_col)
    if added_te:
        p.step(f"Neighbors (train+val→test): {', '.join(sorted(set(added_te)))}")
        for f in added_te:
            if f in trainval_aug.columns and pd.api.types.is_numeric_dtype(trainval_aug[f]):
                if f not in num_cols:
                    num_cols.append(f)

    # Filter usable columns (exist in both train/val and have at least one non-null in train)
    def _refilter(cols: list[str], a: pd.DataFrame, b: pd.DataFrame) -> list[str]:
        return [c for c in cols if (c in a.columns and c in b.columns and a[c].notna().any())]

    num_cols_tr_va = _refilter(num_cols, tr_df_aug, va_df_aug)
    cat_cols_tr_va = _refilter(cat_cols, tr_df_aug, va_df_aug)

    # Also construct trainval/test versions
    num_cols_trv_te = _refilter(num_cols, trainval_aug, te_df_aug)
    cat_cols_trv_te = _refilter(cat_cols, trainval_aug, te_df_aug)

    # Sanity: require at least one feature
    if not (num_cols_tr_va or cat_cols_tr_va):
        raise ValueError("No usable features for train/val. Check features config.")
    if not (num_cols_trv_te or cat_cols_trv_te):
        raise ValueError("No usable features for trainval/test. Check features config.")

    # Attach feature lists (finalized per split)
    tr_df_aug.attrs["num_cols"] = num_cols_tr_va
    tr_df_aug.attrs["cat_cols"] = cat_cols_tr_va
    va_df_aug.attrs["num_cols"] = num_cols_tr_va
    va_df_aug.attrs["cat_cols"] = cat_cols_tr_va

    trainval_aug.attrs["num_cols"] = num_cols_trv_te
    trainval_aug.attrs["cat_cols"] = cat_cols_trv_te
    te_df_aug.attrs["num_cols"] = num_cols_trv_te
    te_df_aug.attrs["cat_cols"] = cat_cols_trv_te

    return tr_df_aug, va_df_aug, te_df_aug, num_cols_tr_va, cat_cols_tr_va, [target_col, use_log]


def _fit_eval_one(
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    use_log: bool,
    device_hint: Optional[str] = None,
) -> tuple[dict, Pipeline, float, np.ndarray]:
    """
    Fit estimator on train, compute Duan smearing on train residuals if log-target,
    evaluate on val (inverse transform + smearing). Return:
      (metrics_val, fitted_pipeline, smear_train, val_pred_array)
    """
    pipe = _build_pipeline(estimator, X_train.attrs["num_cols"], X_train.attrs["cat_cols"], cfg={})
    y_train_fit = np.log1p(y_train.to_numpy(dtype=float)) if use_log else y_train.to_numpy(dtype=float)
    pipe.fit(X_train, y_train_fit)

    # Smearing on TRAIN residuals
    if use_log:
        yhat_tr = pipe.predict(X_train)
        smear_train = _duan_smearing_factor(y_train_fit, yhat_tr)
    else:
        smear_train = None

    # VAL predictions (inverse + smear)
    yhat_val_fit = pipe.predict(X_val)
    yhat_val = _inverse_target(yhat_val_fit, use_log=use_log, smear=smear_train)
    metrics_val = _compute_metrics(y_val.to_numpy(dtype=float), yhat_val)

    return metrics_val, pipe, (smear_train if smear_train is not None else float("nan")), yhat_val

def _fit_eval_one_matrix(
    estimator,
    X_train_mat,
    y_train: pd.Series,
    X_val_mat,
    y_val: pd.Series,
    *,
    use_log: bool,
) -> tuple[dict, float, np.ndarray]:
    """
    Fast path for sweeps: work on pre-encoded matrices instead of re-fitting
    the whole preprocessing pipeline each time.

    Returns:
      metrics_val, smear_train (nan if no log transform), yhat_val (in dollars)
    """
    y_train_arr = y_train.to_numpy(dtype=float)
    if use_log:
        y_train_fit = np.log1p(y_train_arr)
    else:
        y_train_fit = y_train_arr

    # Fit the estimator directly on the encoded matrix
    estimator.fit(X_train_mat, y_train_fit)

    # Duan smearing on TRAIN residuals
    if use_log:
        yhat_tr_fit = estimator.predict(X_train_mat)
        smear_train = _duan_smearing_factor(y_train_fit, yhat_tr_fit)
    else:
        smear_train = None

    # VAL predictions (inverse + smear)
    yhat_val_fit = estimator.predict(X_val_mat)
    yhat_val = _inverse_target(yhat_val_fit, use_log=use_log, smear=smear_train)

    metrics_val = _compute_metrics(y_val.to_numpy(dtype=float), yhat_val)
    return metrics_val, (smear_train if smear_train is not None else float("nan")), yhat_val


def _try_feature_importances(pipe: Pipeline, X_like: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Group OHE back to original categorical columns and sum importances.
    Returns DataFrame ['group','importance'] or None.
    """
    try:
        mdl = pipe.named_steps.get("mdl")
        if mdl is None or not hasattr(mdl, "feature_importances_"):
            return None

        pre_ct = pipe.named_steps.get("pre")
        if pre_ct is None:
            return None

        # fetch cat pipeline + expanded names
        cat_pipeline = None
        cat_cols_used = []
        for name, transf, cols in pre_ct.transformers_:
            if name == "cat":
                cat_pipeline = transf
                cat_cols_used = list(cols)
                break

        if cat_pipeline is None:
            cat_feature_names = []
        else:
            oh = cat_pipeline.named_steps.get("oh")
            if oh is not None:
                cat_feature_names = list(oh.get_feature_names_out(cat_cols_used))
            else:
                cat_feature_names = []

        feat_names = list(X_like.attrs["num_cols"]) + cat_feature_names
        importances = mdl.feature_importances_
        if len(importances) != len(feat_names):
            return pd.DataFrame(
                {"group": [f"f{i}" for i in range(len(importances))], "importance": importances}
            )

        fi = pd.DataFrame({"feature": feat_names, "importance": importances})
        def group_name(s: str) -> str:
            for c in cat_cols_used:
                if s.startswith(c + "_"):
                    return c
            return s
        fi["group"] = fi["feature"].map(group_name)
        return fi.groupby("group", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
    except Exception:
        return None


def run(cfg_path: str = "configs/config.yaml") -> None:
    # ----------------------------------------------------------------------------------
    # Load config and prepare folders
    # ----------------------------------------------------------------------------------
    cfg = _load_yaml(cfg_path)
    if "paths" not in cfg or "data_csv" not in cfg["paths"]:
        raise KeyError("'paths.data_csv' is missing in configs.")

    out_dir = Path(cfg["paths"]["out_dir"])
    models_dir = out_dir / "models"
    metrics_dir = out_dir / "metrics"
    preds_dir = out_dir / "preds"
    figures_dir = out_dir / "figures"
    for d in (models_dir, metrics_dir, preds_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    timestamped = bool(cfg.get("run", {}).get("timestamped_outputs", True))
    run_name = cfg.get("run", {}).get("run_name") or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = out_dir / "run" / run_name if timestamped else None
    if run_dir:
        (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
        (run_dir / "preds").mkdir(parents=True, exist_ok=True)
        (run_dir / "models").mkdir(parents=True, exist_ok=True)

    # Steps counter (rough)
    total_steps = 20
    p = Prog(enabled=True, total=total_steps, desc="Phase 6: Model Sweep")

    # ----------------------------------------------------------------------------------
    # Data, features, split
    # ----------------------------------------------------------------------------------
    tr_df, va_df, te_df, num_cols, cat_cols, tgt = _prep_features_and_split(cfg, p)
    target_col, use_log = tgt[0], bool(tgt[1])
    p.step("Prepared features & finalized splits")


    # Build a single preprocessor and encode TRAIN / VAL once for sweeps
    X_tr_df = tr_df[num_cols + cat_cols]
    X_va_df = va_df[num_cols + cat_cols]
    y_tr = tr_df[target_col]
    y_va = va_df[target_col]

    pre_for_sweep = preprocessor(num_cols, cat_cols, cfg=cfg)
    pre_for_sweep.fit(X_tr_df)

    X_tr_enc = _to_float32(pre_for_sweep.transform(X_tr_df))
    X_va_enc = _to_float32(pre_for_sweep.transform(X_va_df))




    # ----------------------------------------------------------------------------------
    # Baseline (zip×year median) for val & test
    # ----------------------------------------------------------------------------------
    base_val = median_by_zip_year(tr_df, va_df, target=target_col)
    base_test = median_by_zip_year(pd.concat([tr_df, va_df], ignore_index=True), te_df, target=target_col)
    base_metrics_val = _compute_metrics(va_df[target_col].to_numpy(dtype=float), base_val.to_numpy(dtype=float))
    base_metrics_test = _compute_metrics(te_df[target_col].to_numpy(dtype=float), base_test.to_numpy(dtype=float))
    p.step("Computed baselines for val/test")

    # ----------------------------------------------------------------------------------
    # Build grids (overrideable via config.sweep.*)
    # ----------------------------------------------------------------------------------
    sweep_cfg = cfg.get("sweep", {}) or {}
    enable_xgb = bool(sweep_cfg.get("xgb", {}).get("enabled", True))

    # --- cuML banner / toggles ---
    cu_cfg = (sweep_cfg.get("use_cuml", {}) or {})
    cu_global = bool(cu_cfg.get("enabled", False))
    cu_for_ridge = cu_global and bool(cu_cfg.get("ridge", False))
    cu_for_rf = cu_global and bool(cu_cfg.get("rf", False))
    cu_info = _probe_cuml()
    if cu_global:
        print(f"[cuML] requested=True available={cu_info['available']} note={cu_info['note']} ver={cu_info['version']}")
        if not cu_info["available"]:
            print("[cuML] Falling back to sklearn for Ridge/RF.")
    else:
        print("[cuML] requested=False (Ridge/RF default to sklearn)")

    # --- XGBoost GPU probe (mirror other phase scripts) ---
    xgb_device_plan = {"gpu_ok": False, "note": "XGB disabled by config", "gpu_params": {}}
    if enable_xgb:
        gpu_ok, note, gpu_params = _probe_xgb_gpu()
        xgb_device_plan.update({"gpu_ok": gpu_ok, "note": note, "gpu_params": gpu_params})
        print(f"[GPU probe] ok={gpu_ok} note={note}")
        planned = "gpu(auto)" if gpu_ok else "cpu(hist)"
        print(f"[Device] requested=auto  planned={planned}  (Ridge/RF via cuML if enabled)")
    else:
        print("[Device] XGB disabled by config")

    if cu_for_rf and cu_info["available"] and cu_info["rf_cls"] is not None:
        print("[RF backend] cuML (GPU) will be used for RF trials.")
    else:
        if cu_for_rf:
            print(f"[RF backend] cuML requested but unavailable → using sklearn (CPU).  note={cu_info['note']}")
        else:
            print("[RF backend] sklearn (CPU) (cuML not requested)")


    ridge_grid = sweep_cfg.get("ridge", {}).get("alpha", [0.5, 1.0, 2.0, 5.0])

    rf_grid = {
        "n_estimators": sweep_cfg.get("rf", {}).get("n_estimators", [400, 800]),
        "min_samples_leaf": sweep_cfg.get("rf", {}).get("min_samples_leaf", [1, 2, 4]),
        "max_depth": sweep_cfg.get("rf", {}).get("max_depth", [None, 20, 40]),
        "random_state": [int(cfg.get("run", {}).get("random_state", 0))],
        "n_jobs": [-1],
    }

    xgb_grid = {
        "learning_rate": sweep_cfg.get("xgb", {}).get("learning_rate", [0.04, 0.05, 0.06]),
        "max_depth": sweep_cfg.get("xgb", {}).get("max_depth", [6, 8, 10]),
        "n_estimators": sweep_cfg.get("xgb", {}).get("n_estimators", [600, 900, 1200]),
        "subsample": sweep_cfg.get("xgb", {}).get("subsample", [0.8]),
        "colsample_bytree": sweep_cfg.get("xgb", {}).get("colsample_bytree", [0.8]),
        "random_state": [int(cfg.get("run", {}).get("random_state", 0))],
        "verbosity": [0],
        "objective": ["reg:squarederror"],
        # device/tree_method injected after GPU probe
    }

    # ----------------------------------------------------------------------------------
    # Sweep: Ridge
    # ----------------------------------------------------------------------------------
    trials_ridge = []
    # No pipeline stored in best_ridge now; we refit later on train+val
    best_ridge = {"params": None, "metrics_val": None, "smear_train": None, "device_used": None}

    if tqdm is not None:
        tq = tqdm(total=len(ridge_grid), desc="Ridge sweep", leave=True)
    else:
        tq = None

    rs = int(cfg.get("run", {}).get("random_state", 0))
    for alpha in ridge_grid:
        est, dev_hint = _build_ridge(alpha=float(alpha), rs=rs, use_cuml=cu_for_ridge, cu_info=cu_info)

        # Fast path: use pre-encoded matrices instead of refitting preprocessing each time
        metrics_val, smear, _ = _fit_eval_one_matrix(
            est,
            X_tr_enc,
            y_tr,
            X_va_enc,
            y_va,
            use_log=use_log,
        )

        trial = {
            "params": {"alpha": float(alpha)},
            "metrics_val": metrics_val,
            "smear_train": smear,
            "device_used": dev_hint,
        }
        trials_ridge.append(trial)

        if (best_ridge["metrics_val"] is None) or (metrics_val["R2"] > best_ridge["metrics_val"]["R2"]):
            best_ridge.update(
                params=trial["params"],
                metrics_val=metrics_val,
                smear_train=smear,
                device_used=dev_hint,
            )

        if tq:
            tq.set_postfix_str(f"alpha={alpha} R2={metrics_val['R2']:.3f}", refresh=True)
            tq.update(1)

    if tq:
        tq.close()
    p.step("Ridge sweep complete")


    # ----------------------------------------------------------------------------------
    # Sweep: Random Forest
    # ----------------------------------------------------------------------------------
    rf_candidates = []
    for n in rf_grid["n_estimators"]:
        for leaf in rf_grid["min_samples_leaf"]:
            for depth in rf_grid["max_depth"]:
                rf_candidates.append({"n_estimators": int(n), "min_samples_leaf": int(leaf), "max_depth": depth})

    if tqdm is not None:
        tq = tqdm(total=len(rf_candidates), desc="RF sweep", leave=True)
    else:
        tq = None

    trials_rf = []
    best_rf = {"params": None, "metrics_val": None, "smear_train": None, "device_used": None}

    for spec in rf_candidates:
        est, dev_hint = _build_rf(spec, rs=rs, use_cuml=cu_for_rf, cu_info=cu_info)
        if _is_cuml_estimator(est) and "min_samples_leaf" in spec:
            # Warn once that min_samples_leaf is ignored for cuML
            warnings.warn(
                "cuML RF: 'min_samples_leaf' not used; using n_estimators/max_depth.",
                RuntimeWarning,
            )

        # Fast path: use pre-encoded matrices
        metrics_val, smear, _ = _fit_eval_one_matrix(
            est,
            X_tr_enc,
            y_tr,
            X_va_enc,
            y_va,
            use_log=use_log,
        )

        trial = {
            "params": deepcopy(spec),
            "metrics_val": metrics_val,
            "smear_train": smear,
            "device_used": dev_hint,
        }
        trials_rf.append(trial)

        if (best_rf["metrics_val"] is None) or (metrics_val["R2"] > best_rf["metrics_val"]["R2"]):
            best_rf.update(
                params=trial["params"],
                metrics_val=metrics_val,
                smear_train=smear,
                device_used=dev_hint,
            )

        if tq:
            tq.set_postfix_str(f"{spec} R2={metrics_val['R2']:.3f}", refresh=True)
            tq.update(1)

    if tq:
        tq.close()
    p.step("Random Forest sweep complete")

    # ----------------------------------------------------------------------------------
    # Sweep: XGBoost (optional)
    # ----------------------------------------------------------------------------------
    trials_xgb = []
    best_xgb = {"params": None, "metrics_val": None, "pipe": None, "smear_train": None, "device_used": None}

    if enable_xgb:
        try:
            from xgboost import XGBRegressor  # noqa: F401

            gpu_ok = xgb_device_plan["gpu_ok"]
            gpu_params = xgb_device_plan["gpu_params"]

            # Build candidate list
            xgb_candidates = []
            for lr in xgb_grid["learning_rate"]:
                for md in xgb_grid["max_depth"]:
                    for ne in xgb_grid["n_estimators"]:
                        for ss in xgb_grid["subsample"]:
                            for cs in xgb_grid["colsample_bytree"]:
                                cand = {
                                    "learning_rate": float(lr),
                                    "max_depth": int(md),
                                    "n_estimators": int(ne),
                                    "subsample": float(ss),
                                    "colsample_bytree": float(cs),
                                    "random_state": rs,
                                    "verbosity": 0,
                                    "objective": "reg:squarederror",
                                }
                                if gpu_ok:
                                    cand.update(gpu_params)  # e.g. {'device':'cuda','tree_method':'hist','predictor':'gpu_predictor'}
                                else:
                                    cand["tree_method"] = "hist"
                                    cand.pop("predictor", None)
                                    cand.pop("device", None)
                                xgb_candidates.append(cand)

            if tqdm is not None:
                tq = tqdm(total=len(xgb_candidates), desc="XGB sweep", leave=True)
            else:
                tq = None

            for spec in xgb_candidates:
                est = XGBRegressor(**spec)
                metrics_val, pipe, smear, _ = _fit_eval_one(est, X_tr, y_tr, X_va, y_va, use_log=use_log)
                # Determine runtime device (post-fit)
                dev_rt = _xgb_runtime_device(pipe)
                used_device = "gpu" if (dev_rt in ("gpu", "cuda")) else "cpu"
                trial = {
                    "params": {k: v for k, v in spec.items() if k not in ("verbosity",)},
                    "metrics_val": metrics_val,
                    "smear_train": smear,
                    "device_used": used_device,
                }
                trials_xgb.append(trial)
                if (best_xgb["metrics_val"] is None) or (metrics_val["R2"] > best_xgb["metrics_val"]["R2"]):
                    best_xgb.update(
                        params=trial["params"], metrics_val=metrics_val, pipe=pipe, smear_train=smear,
                        device_used=used_device
                    )
                if tq:
                    tq.set_postfix_str(
                        f"md={spec['max_depth']}, ne={spec['n_estimators']}, lr={spec['learning_rate']} R2={metrics_val['R2']:.3f}",
                        refresh=True
                    )
                    tq.update(1)
            if tq:
                tq.close()
        except Exception as e:
            enable_xgb = False
            xgb_device_plan["note"] = f"XGB disabled due to import/fit error: {e}"
    else:
        xgb_device_plan["note"] = "XGB disabled by config"
    p.step("XGBoost sweep complete" if enable_xgb else "XGBoost skipped/disabled")

    # ----------------------------------------------------------------------------------
    # Choose best-per-family and best-overall on VAL R2
    # ----------------------------------------------------------------------------------
    family_results = []
    if best_ridge["metrics_val"] is not None:
        family_results.append(("ridge", best_ridge))
    if best_rf["metrics_val"] is not None:
        family_results.append(("rf", best_rf))
    if enable_xgb and best_xgb["metrics_val"] is not None:
        family_results.append(("xgb", best_xgb))

    if not family_results:
        raise RuntimeError("No successful model trials were completed.")

    def _r2_of(rec): return rec[1]["metrics_val"]["R2"]
    best_overall_family, best_overall = max(family_results, key=_r2_of)

    # ----------------------------------------------------------------------------------
    # Final refits on TRAIN+VAL and evaluate on TEST
    # ----------------------------------------------------------------------------------
    trainval_df = pd.concat([tr_df, va_df], ignore_index=True)
    X_trv = trainval_df[num_cols + cat_cols]
    X_te = te_df[num_cols + cat_cols]
    y_trv = trainval_df[target_col]
    y_te = te_df[target_col]

    final_artifacts = {}
    for fam, best in (("ridge", best_ridge), ("rf", best_rf), ("xgb", best_xgb)):
        if fam == "xgb" and (not enable_xgb or best["metrics_val"] is None):
            continue

        # Rebuild estimator from best params
        if fam == "ridge":
            est, dev_hint = _build_ridge(alpha=float(best["params"]["alpha"]), rs=rs, use_cuml=cu_for_ridge, cu_info=cu_info)
        elif fam == "rf":
            est, dev_hint = _build_rf(best["params"], rs=rs, use_cuml=cu_for_rf, cu_info=cu_info)
        else:  # xgb
            from xgboost import XGBRegressor
            est = XGBRegressor(**best["params"])
            dev_hint = None  # we will infer

        pipe = _build_pipeline(est, X_trv.attrs["num_cols"], X_trv.attrs["cat_cols"], cfg={})
        y_trv_fit = np.log1p(y_trv.to_numpy(dtype=float)) if use_log else y_trv.to_numpy(dtype=float)
        pipe.fit(X_trv, y_trv_fit)

        smear_trv = None
        if use_log:
            yhat_trv_fit = pipe.predict(X_trv)
            smear_trv = _duan_smearing_factor(y_trv_fit, yhat_trv_fit)

        yhat_te_fit = pipe.predict(X_te)
        yhat_te = _inverse_target(yhat_te_fit, use_log=use_log, smear=smear_trv)
        metrics_te = _compute_metrics(y_te.to_numpy(dtype=float), yhat_te)

        # Save artifacts
        fam_tag = {"ridge": "ridge", "rf": "rf", "xgb": "xgb"}[fam]
        model_path = (models_dir / f"phase6_{fam_tag}.pkl")
        joblib_dump(pipe, model_path)

        preds = te_df[["sold_date", "city", "state", "zip_code", target_col]].copy()
        preds[f"pred_{fam_tag}"] = yhat_te
        preds[f"pred_baseline"] = base_test.values
        preds_path = preds_dir / f"test_preds_{fam_tag}.csv"
        preds.to_csv(preds_path, index=False)

        fi_path = None
        fi_df = _try_feature_importances(pipe, X_trv)
        if fi_df is not None:
            fi_path = models_dir / f"feature_importances_{fam_tag}.csv"
            fi_df.to_csv(fi_path, index=False)

        used_dev = dev_hint
        if fam == "xgb":
            dev_rt = _xgb_runtime_device(pipe)
            used_dev = "gpu" if dev_rt in ("gpu", "cuda") else "cpu"
            if xgb_device_plan["gpu_ok"] and used_dev != "gpu":
                print("[WARN] Final XGB model is running on CPU. Check CUDA install & xgboost build.")

        final_artifacts[fam_tag] = {
            "final_metrics_test": metrics_te,
            "smear_trainval": (smear_trv if smear_trv is not None else float("nan")),
            "model_path": str(model_path),
            "preds_path": str(preds_path),
            "feature_importances_path": (str(fi_path) if fi_path else None),
            "final_model_params": _safe_model_params(pipe),
            "device_used_final": used_dev,
        }
    p.step("Final refits on train+val and test evaluation complete")

    # ----------------------------------------------------------------------------------
    # Consolidated metrics JSON
    # ----------------------------------------------------------------------------------
    run_info = {
        "phase": "6",
        "run_id": run_name,
        "random_state": cfg.get("run", {}).get("random_state", 0),
        "data_csv": str(cfg["paths"]["data_csv"]),
        "data_hash": _sha256_file(Path(cfg["paths"]["data_csv"])),
        "train_rows": int(len(tr_df)),
        "val_rows": int(len(va_df)),
        "test_rows": int(len(te_df)),
        "split_years": {
            "train_max": int(cfg.get("split", {}).get("train_max_year", 2022)),
            "val": int(cfg.get("split", {}).get("val_year", 2023)),
            "test_min": int(cfg.get("split", {}).get("test_min_year", 2024)),
        },
        "feature_counts": {"numeric": len(num_cols), "categorical": len(cat_cols)},
        "gpu_probe": None,
        "cuml": {
            "requested": {"enabled": cu_global, "ridge": cu_for_ridge, "rf": cu_for_rf},
            "available": cu_info["available"],
            "version": cu_info["version"],
            "note": cu_info["note"],
        },
    }
    if enable_xgb:
        run_info["gpu_probe"] = {
            "ok": xgb_device_plan["gpu_ok"],
            "note": xgb_device_plan["note"],
            "injected_params": xgb_device_plan["gpu_params"],
        }

    payload = {
        "run_info": run_info,
        "target": {"name": target_col, "log_transform": use_log},
        "baseline": {
            "val": base_metrics_val,
            "test": base_metrics_test,
        },
        "ridge": {
            "trials": trials_ridge,
            "best_val": {
                "params": best_ridge["params"],
                "metrics_val": best_ridge["metrics_val"],
                "smear_train": (best_ridge["smear_train"] if best_ridge["smear_train"] is not None else float("nan")),
                "device_used": best_ridge["device_used"],
            },
        },
        "rf": {
            "trials": trials_rf,
            "best_val": {
                "params": best_rf["params"],
                "metrics_val": best_rf["metrics_val"],
                "smear_train": (best_rf["smear_train"] if best_rf["smear_train"] is not None else float("nan")),
                "device_used": best_rf["device_used"],
            },
        },
        "xgb": None,  # filled if enabled
        "best_overall_on_val": {
            "family": best_overall_family,
            "metrics_val": best_overall["metrics_val"],
            "params": best_overall["params"],
        },
        "final": final_artifacts,  # per-family test results + paths
    }
    if enable_xgb:
        payload["xgb"] = {
            "trials": trials_xgb,
            "best_val": {
                "params": best_xgb["params"],
                "metrics_val": best_xgb["metrics_val"],
                "device_used": best_xgb.get("device_used"),
                "smear_train": (best_xgb["smear_train"] if best_xgb["smear_train"] is not None else float("nan")),
            },
        }

    # Write metrics
    metrics_path = metrics_dir / "phase6_sweep_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2))
    if run_dir:
        (run_dir / "metrics" / "phase6_sweep_metrics.json").write_text(json.dumps(payload, indent=2))

    # Save config snapshot for provenance
    if bool(cfg.get("run", {}).get("save_config_snapshot", True)) and run_dir:
        (run_dir / "metrics" / "config.snapshot.yaml").write_text(Path(cfg_path).read_text())
    p.step("Saved metrics and config snapshot")

    p.close()
    # Pretty-print the high-level outcome to stdout
    print(json.dumps(
        {
            "run_info": {
                "run_id": run_name,
                "train_rows": int(len(tr_df)),
                "val_rows": int(len(va_df)),
                "test_rows": int(len(te_df)),
            },
            "baseline": {"val": base_metrics_val, "test": base_metrics_test},
            "best_overall_on_val": {
                "family": best_overall_family,
                "metrics_val": best_overall["metrics_val"],
                "params": best_overall["params"],
            },
            "final_test_metrics_by_family": {
                fam: final_artifacts[fam]["final_metrics_test"]
                for fam in final_artifacts
            },
        },
        indent=2
    ))


if __name__ == "__main__":
    run()
