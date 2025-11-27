# src/realestate/train_phase6.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Mapping, Iterable
from datetime import datetime
from copy import deepcopy
import hashlib

import numpy as np
import pandas as pd
import yaml
from joblib import dump as joblib_dump
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import io as io_mod
from .features import (
    basic_time_feats,
    years_since_prev,
    apply_engineered,
    maybe_add_neighbors_via_cfg,
)
from .baselines import median_by_zip_year
from .modeling import preprocessor, make_model, describe_estimator

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


# ----------------------------- #
# Tiny progress helper
# ----------------------------- #
class Prog:
    def __init__(self, enabled: bool, total: int, desc: str):
        self.enabled = enabled and (tqdm is not None)
        self.t = tqdm(total=total, desc=desc, leave=True) if self.enabled else None

    def step(self, msg: str):
        if self.t:
            self.t.set_postfix_str(msg[:70], refresh=True)
            self.t.update(1)

    def close(self):
        if self.t:
            self.t.close()


# ----------------------------- #
# Time-based splitting
# ----------------------------- #
def _require_dt_sold_date(df: pd.DataFrame):
    if "sold_date" not in df.columns or not pd.api.types.is_datetime64_any_dtype(df["sold_date"]):
        raise ValueError(
            "Phase 6 requires a real datetime 'sold_date' column for year-based splitting.\n"
            "Ensure io.load_data() parses 'sold_date' to datetime."
        )


def tri_temporal_split_by_year(
    df: pd.DataFrame,
    *,
    train_max_year: int = 2022,
    val_year: int = 2023,
    test_min_year: int = 2024,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """train: year <= train_max_year; val: year == val_year; test: year >= test_min_year."""
    _require_dt_sold_date(df)
    yrs = df["sold_date"].dt.year
    train_df = df.loc[yrs <= train_max_year].copy()
    val_df = df.loc[yrs == val_year].copy()
    test_df = df.loc[yrs >= test_min_year].copy()
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError(
            f"Empty split: train(len={len(train_df)}), val(len={len(val_df)}), test(len={len(test_df)}). "
            f"Check your data coverage."
        )
    return train_df, val_df, test_df


# ----------------------------- #
# XGBoost GPU probing & runtime device
# ----------------------------- #
def _probe_xgb_gpu() -> tuple[bool, str, dict]:
    """
    Return (ok, note, suggested_params_for_gpu).
    Confirms that XGBoost actually executed on GPU. Avoids using legacy
    'gpu_hist' on XGBoost >= 2.0 where it's invalid.
    """
    try:
        import xgboost as xgb
    except Exception as e:
        return False, f"xgboost import failed: {e}", {}

    ver_str = getattr(xgb, "__version__", "unknown")
    try:
        major = int(str(ver_str).split(".")[0])
    except Exception:
        major = 0

    rng = np.random.default_rng(0)
    X = rng.normal(size=(32, 8)).astype(np.float32)
    y = rng.normal(size=(32,)).astype(np.float32)
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
    if major >= 2:
        try:
            params = {"device": "cuda", "tree_method": "hist", "verbosity": 0}
            booster = xgb.train(params, dtrain, num_boost_round=1)
            used = _runtime_device(booster)
            if used in ("cuda", "gpu"):
                return True, f"xgboost {ver_str} (device=cuda OK)", params
            notes.append(f"device=cuda accepted but used={used}")
        except Exception as e:
            notes.append(f"device=cuda failed: {e}")
    else:
        try:
            params = {"tree_method": "gpu_hist", "predictor": "gpu_predictor", "verbosity": 0}
            booster = xgb.train(params, dtrain, num_boost_round=1)
            used = _runtime_device(booster)
            if used in ("cuda", "gpu"):
                return True, f"xgboost {ver_str} (gpu_hist OK)", params
            notes.append(f"gpu_hist accepted but used={used}")
        except Exception as e:
            notes.append(f"gpu_hist failed: {e}")

    return False, f"xgboost {ver_str} GPU probe failed; " + " | ".join(notes), {}


def _xgb_runtime_device(pipe) -> str:
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


# ----------------------------- #
# Metrics / utilities
# ----------------------------- #
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
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return f"sha256:{h.hexdigest()}"


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Mapping[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def _duan_smearing_factor(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """
    Duan (1983) smearing factor for log-link models.
    We use log1p / expm1 convention consistently.
    """
    resid = y_true_log - y_pred_log
    return float(np.mean(np.exp(resid)))


def _inverse_log_with_smear(y_pred_log: np.ndarray, smear: float) -> np.ndarray:
    return np.expm1(y_pred_log) * smear


def _ensure_feature_lists(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feats_cfg: Mapping[str, Iterable[str]],
    added_numeric: Iterable[str],
) -> tuple[list[str], list[str]]:
    all_cols = set(df_train.columns) & set(df_val.columns) & set(df_test.columns)
    num_cfg = [c for c in (feats_cfg.get("numeric") or []) if c in all_cols]
    cat_cfg = [c for c in (feats_cfg.get("categorical") or []) if c in all_cols]

    # add neighbor numeric feats that landed from builders
    for f in added_numeric:
        if f in all_cols and f not in num_cfg and pd.api.types.is_numeric_dtype(df_train[f]):
            num_cfg.append(f)

    # keep only columns that have at least one non-null in TRAIN
    num_cols = [c for c in num_cfg if df_train[c].notna().any()]
    cat_cols = [c for c in cat_cfg if df_train[c].notna().any()]

    if not (num_cols or cat_cols):
        raise ValueError("No usable features remain. Check configs.features and neighbor settings.")
    return num_cols, cat_cols


# ----------------------------- #
# Safe, compact model params (v2)
# ----------------------------- #
def _safe_model_params(pipe) -> dict[str, Any]:
    """
    Extract model params for logging, but make them JSON-safe and compact:
      - Convert NumPy/pandas scalars to Python types.
      - Replace NaN/Inf with None.
      - Summarize large arrays/dataframes/long sequences and huge dicts.
      - Truncate very long strings.
      - Avoid dumping nested estimators or giant objects.
    """
    MAX_STR = 300
    MAX_SEQ_ITEMS = 25
    MAX_DICT_ITEMS = 30

    def _is_nan(x) -> bool:
        try:
            return bool(np.isnan(x))
        except Exception:
            return False

    def _is_inf(x) -> bool:
        try:
            return bool(np.isinf(x))
        except Exception:
            return False

    def _to_native_scalar(x):
        if isinstance(x, np.generic):
            return x.item()
        if "pandas" in str(type(x)):
            try:
                import pandas as pd
                if isinstance(x, pd.Timestamp):
                    return x.isoformat()
                if isinstance(x, pd.Timedelta):
                    return str(x)
            except Exception:
                pass
        return x

    def _summarize_obj(x, depth: int = 0):
        x = _to_native_scalar(x)
        if x is None or isinstance(x, (bool, int)):
            return x
        if isinstance(x, float):
            if _is_nan(x) or _is_inf(x):
                return None
            return float(x)
        if isinstance(x, str):
            return x if len(x) <= MAX_STR else (x[:MAX_STR] + "…")
        if isinstance(x, np.ndarray):
            try:
                return f"ndarray(shape={x.shape}, dtype={x.dtype})"
            except Exception:
                return "ndarray"
        try:
            import pandas as pd
            if isinstance(x, (pd.Series, pd.Index)):
                n = len(x)
                if n <= MAX_SEQ_ITEMS:
                    return [_summarize_obj(v, depth + 1) for v in x.tolist()]
                return f"{type(x).__name__}(len={n}, dtype={getattr(x, 'dtype', 'unknown')})"
            if isinstance(x, pd.DataFrame):
                return f"DataFrame(shape={x.shape}, columns={list(x.columns)[:8]}{'…' if x.shape[1] > 8 else ''})"
        except Exception:
            pass
        if isinstance(x, (list, tuple, set)):
            seq = list(x)
            n = len(seq)
            if n <= MAX_SEQ_ITEMS:
                return [_summarize_obj(v, depth + 1) for v in seq]
            return f"{type(x).__name__}(len={n})"
        if isinstance(x, dict):
            items = list(x.items())
            n = len(items)
            if n <= MAX_DICT_ITEMS:
                out = {}
                for k, v in items:
                    out[str(k)] = _summarize_obj(v, depth + 1)
                return out
            try:
                preview_keys = [str(k) for k, _ in items[:8]]
            except Exception:
                preview_keys = []
            return {"__summary__": f"dict(len={n})", "__preview_keys__": preview_keys}
        if callable(x):
            try:
                return f"<callable {getattr(x, '__name__', type(x).__name__)}>"
            except Exception:
                return "<callable>"
        if hasattr(x, "__module__") and hasattr(x, "__class__"):
            try:
                r = repr(x)
                return r if len(r) <= MAX_STR else f"<{x.__class__.__name__}>"
            except Exception:
                return f"<{x.__class__.__name__}>"
        try:
            r = repr(x)
            return r if len(r) <= MAX_STR else (r[:MAX_STR] + "…")
        except Exception:
            return f"<unrepr {type(x).__name__}>"

    try:
        mdl = pipe.named_steps.get("mdl", None)
        if mdl is None:
            return {}
        raw = mdl.get_params(deep=False)
        cleaned = {}
        for k, v in raw.items():
            cv = _summarize_obj(v, 0)
            if isinstance(cv, float) and (np.isnan(cv) or np.isinf(cv)):
                cv = None
            cleaned[str(k)] = cv
        for noisy in ("feature_weights", "feature_types"):
            if noisy in cleaned and not isinstance(cleaned[noisy], (str, int, float, bool, type(None))):
                cleaned[noisy] = str(type(cleaned[noisy]).__name__)
        return {k: cleaned[k] for k in sorted(cleaned)}
    except Exception:
        return {}


# ----------------------------- #
# Phase 6 Entrypoint
# ----------------------------- #
def run(cfg_path: str = "configs/config.yaml") -> None:
    cfg = _load_yaml(cfg_path)

    # Steps accounting (roughly)
    timestamped = bool(cfg.get("run", {}).get("timestamped_outputs", True))
    save_pipeline = bool(cfg.get("train", {}).get("save_pipeline", True))
    save_fi = bool(cfg.get("train", {}).get("save_feature_importances", True))
    total_steps = 0
    total_steps += 16  # dirs, run folder, load, feats, split, neighbors, features, baselines, pre, plan, search, refit, preds, metrics, save, snapshot
    if save_pipeline:
        total_steps += 1
    if save_fi:
        total_steps += 1
    if timestamped:
        total_steps += 4
    p = Prog(enabled=True, total=total_steps, desc="Phase 6: Train/Val/Test")

    # Paths
    if "paths" not in cfg or "data_csv" not in cfg["paths"]:
        raise KeyError("'paths.data_csv' missing in config.")
    data_csv = Path(cfg["paths"]["data_csv"])
    out_dir = Path(cfg["paths"]["out_dir"])
    models_dir = out_dir / "models"
    metrics_dir = out_dir / "metrics"
    preds_dir = out_dir / "preds"
    figures_dir = out_dir / "figures"
    for q in (models_dir, metrics_dir, preds_dir, figures_dir):
        q.mkdir(parents=True, exist_ok=True)
    p.step("Prepared output dirs")

    # Run folder
    run_name = cfg.get("run", {}).get("run_name") or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = out_dir / "run" / run_name if timestamped else None
    if run_dir:
        for sub in ("metrics", "preds", "models"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
        p.step(f"Run folder: {run_name}")
    else:
        p.step("Run folder disabled")

    # Load + features
    df = io_mod.load_data(data_csv, cfg=cfg)
    p.step("Loaded CSV + schema snapshot")
    _require_dt_sold_date(df)

    # Row-wise features (time)
    df = basic_time_feats(df)
    df = years_since_prev(df)
    p.step("Derived features (time)")
    df = apply_engineered(df, cfg)
    p.step("Applied engineered features")

    # Split 3-way by year
    train_df, val_df, test_df = tri_temporal_split_by_year(df, train_max_year=2022, val_year=2023, test_min_year=2024)
    p.step("3-way temporal split complete")

    # Target config
    target_col = cfg.get("target", {}).get("name", "price")
    use_log = bool(cfg.get("target", {}).get("log_transform", True))
    if target_col not in train_df.columns:
        raise KeyError(f"Target column '{target_col}' not found.")

    # Leakage-safe neighbors:
    #   Build val features from train; later rebuild test features from train+val (post-refit).
    added_val_feats: list[str] = []
    train_for_val, val_aug, added_val_feats = maybe_add_neighbors_via_cfg(
        train_df.copy(), val_df.copy(), cfg, target=target_col
    )
    # We don't mutate 'train_df' in-place above; ensure train_df == train_for_val (it may be enriched, keep it)
    train_df = train_for_val
    val_df = val_aug
    p.step(f"Neighbors for val: +{len(added_val_feats)} feats" if added_val_feats else "No neighbor feats for val")

    # Feature lists (preliminary)
    feats_cfg = cfg.get("features", {}) or {}
    num_cols, cat_cols = _ensure_feature_lists(train_df, val_df, test_df, feats_cfg, added_val_feats)
    p.step(f"Features selected: num={len(num_cols)} cat={len(cat_cols)}")

    # Baselines (per-slice)
    base_val = median_by_zip_year(train_df, val_df, target=target_col)
    base_test_from_train = median_by_zip_year(train_df, test_df, target=target_col)  # baseline doesn't get val info
    p.step("Baselines computed")

    # Preprocessor
    pre = preprocessor(num_cols, cat_cols, cfg=cfg)

    # ---- Device & backend selection (GPU-aware for XGBoost 1.x/2.x)
    requested_device = str(cfg.get("run", {}).get("device", "auto")).lower()
    resolved_device = "cpu"
    auto_switched = False
    kind = cfg.get("model", {}).get("kind", "random_forest")

    gpu_ok, gpu_note, gpu_params = (False, "", {})
    if requested_device in ("gpu", "auto"):
        gpu_ok, gpu_note, gpu_params = _probe_xgb_gpu()

    if gpu_ok:
        resolved_device = "gpu" if requested_device == "gpu" else "gpu(auto)"
        if kind != "xgboost":
            kind = "xgboost"
            cfg.setdefault("model", {})["kind"] = "xgboost"
            auto_switched = True

    if kind == "xgboost":
        xgb_cfg = cfg.setdefault("model", {}).setdefault("xgboost", {})
        if gpu_ok:
            xgb_cfg.update(gpu_params)  # device='cuda' + 'hist' for ≥2.0
        else:
            xgb_cfg.clear()
            xgb_cfg["tree_method"] = "hist"
            xgb_cfg.pop("predictor", None)
            xgb_cfg.pop("device", None)
            resolved_device = "cpu"

    # Announce plan
    lines = []
    lines.append(f"[GPU probe] ok={gpu_ok} note={gpu_note}")
    suffix = " (auto-switch → xgboost)" if auto_switched else ""
    lines.append(f"[Device] requested={requested_device}  planned={resolved_device}{suffix}")
    msg = "\n".join(lines)
    if tqdm is not None:
        tqdm.write("")
        tqdm.write(msg)
    else:
        print("\n" + msg, flush=True)

    # ----------------------------- #
    # Hyperparam search on validation
    # ----------------------------- #
    # Build initial model (may be changed by search loop)
    base_pipe = make_model(kind, pre, cfg=cfg)

    # Candidate grid (override via cfg['tune']['grid'] if present)
    default_grid = [
        {"n_estimators": 600,  "max_depth": 6,  "learning_rate": 0.06, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 900,  "max_depth": 8,  "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 1200, "max_depth": 8,  "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 1200, "max_depth": 10, "learning_rate": 0.04, "subsample": 0.8, "colsample_bytree": 0.8},
    ]
    grid = cfg.get("tune", {}).get("grid", default_grid)

    # Prepare training data for search (train slice only)
    y_train = train_df[target_col].astype(float).to_numpy()
    y_val = val_df[target_col].astype(float).to_numpy()
    if use_log:
        y_train_fit = np.log1p(y_train)
    else:
        y_train_fit = y_train

    X_train = train_df[num_cols + cat_cols]
    X_val = val_df[num_cols + cat_cols]

    best = {"params": None, "metrics_val": None, "smear_train": None, "used_device": "cpu"}
    tried: list[dict[str, Any]] = []

    for i, candidate in enumerate(grid, 1):
        # Clone a fresh pipeline each try
        pipe = make_model(kind, pre, cfg=cfg)
        # Gently set/override top-level model params (only if model exposes them)
        mdl = pipe.named_steps.get("mdl", None)
        if mdl is not None:
            mdl.set_params(**{k: v for k, v in candidate.items() if k in mdl.get_params(deep=True)})

        # Fit on TRAIN only (no leakage)
        try:
            pipe.fit(X_train, y_train_fit)
        except Exception:
            tm = str(cfg.get("model", {}).get("xgboost", {}).get("tree_method", "")).lower()
            dev_flag = str(cfg.get("model", {}).get("xgboost", {}).get("device", "")).lower()
            if kind == "xgboost" and ("gpu" in tm or dev_flag in ("cuda", "gpu")):
                cfg["model"]["xgboost"]["tree_method"] = "hist"
                cfg["model"]["xgboost"].pop("predictor", None)
                cfg["model"]["xgboost"].pop("device", None)
                pipe = make_model(kind, pre, cfg=cfg)
                mdl = pipe.named_steps.get("mdl", None)
                if mdl is not None:
                    mdl.set_params(**{k: v for k, v in candidate.items() if k in mdl.get_params(deep=True)})
                pipe.fit(X_train, y_train_fit)
            else:
                raise

        # Device actually used
        if kind == "xgboost":
            rt_dev = _xgb_runtime_device(pipe)
            used_dev = "gpu" if (rt_dev == "gpu" or str(rt_dev).startswith("cuda")) else "cpu"
        else:
            used_dev = "cpu"

        # Duan smearing from TRAIN residuals of this fit
        if use_log:
            pred_train_log = pipe.predict(X_train)
            smear_train = _duan_smearing_factor(y_train_fit, pred_train_log)
            pred_val_log = pipe.predict(X_val)
            pred_val = _inverse_log_with_smear(pred_val_log, smear_train)
        else:
            smear_train = 1.0
            pred_val = pipe.predict(X_val)

        # Metrics vs validation ground truth
        metrics_val = _compute_metrics(y_val, pred_val)

        tried.append(
            {
                "index": i,
                "candidate": candidate,
                "metrics_val": metrics_val,
                "smear_train": smear_train,
                "device_used": used_dev,
            }
        )

        # Track best by RMSE (primary), break ties by MAE
        if best["metrics_val"] is None:
            best = {"params": candidate, "metrics_val": metrics_val, "smear_train": smear_train, "used_device": used_dev}
        else:
            cur, bst = metrics_val, best["metrics_val"]
            better = (cur["RMSE"] < bst["RMSE"]) or (np.isclose(cur["RMSE"], bst["RMSE"]) and cur["MAE"] < bst["MAE"])
            if better:
                best = {"params": candidate, "metrics_val": metrics_val, "smear_train": smear_train, "used_device": used_dev}

    p.step(f"Grid search completed ({len(tried)} trials)")

    # ----------------------------- #
    # Rebuild neighbors for TEST using TRAIN+VAL, refit final, evaluate TEST
    # ----------------------------- #
    trainval_df = pd.concat([train_df, val_df], axis=0, ignore_index=True).sort_values("sold_date")
    trainval_for_test, test_aug, added_test_feats = maybe_add_neighbors_via_cfg(
        trainval_df.copy(), test_df.copy(), cfg, target=target_col
    )
    trainval_df = trainval_for_test
    test_df = test_aug
    p.step(f"Neighbors for test: +{len(added_test_feats)} feats" if added_test_feats else "No neighbor feats for test")

    # Final feature lists after potential new neighbor feats
    num_cols_final, cat_cols_final = _ensure_feature_lists(trainval_df, val_df, test_df, feats_cfg, added_test_feats)
    pre_final = preprocessor(num_cols_final, cat_cols_final, cfg=cfg)

    # Final model with best params
    final_pipe = make_model(kind, pre_final, cfg=cfg)
    final_mdl = final_pipe.named_steps.get("mdl", None)
    if final_mdl is not None and best["params"] is not None:
        final_mdl.set_params(**{k: v for k, v in best["params"].items() if k in final_mdl.get_params(deep=True)})

    # Prepare arrays
    y_trainval = trainval_df[target_col].astype(float).to_numpy()
    y_test = test_df[target_col].astype(float).to_numpy()
    y_trainval_fit = np.log1p(y_trainval) if use_log else y_trainval

    X_trainval = trainval_df[num_cols_final + cat_cols_final]
    X_test = test_df[num_cols_final + cat_cols_final]

    # Fit on TRAIN+VAL, evaluate on TEST
    final_pipe.fit(X_trainval, y_trainval_fit)
    if kind == "xgboost":
        rt_dev_final = _xgb_runtime_device(final_pipe)
        used_device_final = "gpu" if (rt_dev_final == "gpu" or str(rt_dev_final).startswith("cuda")) else "cpu"
    else:
        used_device_final = "cpu"

    if use_log:
        pred_trainval_log = final_pipe.predict(X_trainval)
        smear_final = _duan_smearing_factor(y_trainval_fit, pred_trainval_log)
        pred_test_log = final_pipe.predict(X_test)
        pred_test = _inverse_log_with_smear(pred_test_log, smear_final)
    else:
        smear_final = 1.0
        pred_test = final_pipe.predict(X_test)

    # Metrics
    metrics_val_model = best["metrics_val"]
    metrics_val_base = _compute_metrics(y_val, base_val.to_numpy(dtype=float))
    metrics_test_model = _compute_metrics(y_test, pred_test)
    metrics_test_base = _compute_metrics(y_test, base_test_from_train.to_numpy(dtype=float))
    p.step("Metrics computed (val + test)")

    # ---- Output artifacts
    # Predictions CSVs
    keep_cols = [c for c in ["sold_date", "city", "state", "zip_code", target_col] if c in val_df.columns]
    val_preds_df = val_df[keep_cols].copy()
    val_preds_df["pred_baseline"] = base_val.values
    # For val model predictions, we must recompute with the *best trial* pipe to keep consistency:
    # (re-fit the best-trial model again on TRAIN to materialize preds for CSV)
    best_pipe = make_model(kind, pre, cfg=cfg)
    best_mdl = best_pipe.named_steps.get("mdl", None)
    if best_mdl is not None and best["params"] is not None:
        best_mdl.set_params(**{k: v for k, v in best["params"].items() if k in best_mdl.get_params(deep=True)})
    best_pipe.fit(X_train, y_train_fit)
    if use_log:
        val_pred_log_for_csv = best_pipe.predict(X_val)
        val_pred_for_csv = _inverse_log_with_smear(val_pred_log_for_csv, best["smear_train"])
    else:
        val_pred_for_csv = best_pipe.predict(X_val)
    val_preds_df["pred_model"] = val_pred_for_csv

    val_path = preds_dir / "val_preds.csv"
    val_preds_df.to_csv(val_path, index=False)

    keep_cols_t = [c for c in ["sold_date", "city", "state", "zip_code", target_col] if c in test_df.columns]
    test_preds_df = test_df[keep_cols_t].copy()
    test_preds_df["pred_baseline"] = base_test_from_train.values
    test_preds_df["pred_model"] = pred_test
    test_path = preds_dir / "test_preds.csv"
    test_preds_df.to_csv(test_path, index=False)
    p.step("Saved predictions (val & test)")

    # Feature importances (final model)
    fi_df = _try_feature_importances(final_pipe, pre_final, num_cols_final, cat_cols_final)
    if fi_df is not None:
        fi_path = models_dir / "feature_importances_phase6.csv"
        fi_df.to_csv(fi_path, index=False)
        p.step("Saved feature importances")
    else:
        p.step("No feature importances available")

    # Save model (final)
    if bool(cfg.get("train", {}).get("save_pipeline", True)):
        model_path = models_dir / cfg.get("train", {}).get("model_filename", "price_model_phase6.pkl")
        joblib_dump(final_pipe, model_path)
        p.step("Saved final pipeline")

    # Run info + metrics bundle
    run_info = {
        "phase": "6",
        "run_id": run_name,
        "random_state": cfg.get("run", {}).get("random_state", 0),
        "data_csv": str(data_csv),
        "data_hash": _sha256_file(data_csv),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "trainval_rows": int(len(trainval_df)),
        "test_rows": int(len(test_df)),
        "split_years": {"train_max": 2022, "val": 2023, "test_min": 2024},
        "feature_counts": {"numeric": len(num_cols_final), "categorical": len(cat_cols_final)},
        "model_kind": kind,
        "log_target": use_log,
        "device_requested": requested_device,
        "device_used_val_best": best["used_device"],
        "device_used_final": used_device_final,
    }

    payload = {
        "run_info": run_info,
        "target": {
            "name": target_col,
            "log_transform": use_log,
            "duan_smearing_val_best": float(best["smear_train"]) if use_log else 1.0,
            "duan_smearing_final": float(smear_final) if use_log else 1.0,
        },
        "tuning": {
            "trials": tried,
            "best_params": best["params"],
            "best_val_metrics": metrics_val_model,
        },
        "val": {
            "baseline": metrics_val_base,
            "model": metrics_val_model,
        },
        "test": {
            "baseline": metrics_test_base,
            "model": metrics_test_model,
        },
        "final_model_params": _safe_model_params(final_pipe),
    }

    metrics_path = metrics_dir / "metrics_phase6.json"
    metrics_path.write_text(json.dumps(payload, indent=2))
    p.step("Saved metrics_phase6.json")

    # Copy to run/
    if run_dir:
        (run_dir / "metrics" / "metrics_phase6.json").write_text(json.dumps(payload, indent=2))
        val_preds_df.to_csv(run_dir / "preds" / "val_preds.csv", index=False)
        test_preds_df.to_csv(run_dir / "preds" / "test_preds.csv", index=False)
        joblib_dump(final_pipe, run_dir / "models" / (models_dir / cfg.get("train", {}).get("model_filename", "price_model_phase6.pkl")).name)
        if fi_df is not None:
            fi_df.to_csv(run_dir / "models" / "feature_importances_phase6.csv", index=False)
        p.step("Copied artifacts to run/")

    # Config snapshot
    if bool(cfg.get("run", {}).get("save_config_snapshot", True)) and run_dir:
        snap_path = run_dir / "metrics" / "config.snapshot.yaml"
        snap_path.write_text(Path(cfg_path).read_text())
        p.step("Saved config snapshot")

    p.close()

    # Final console summary (short)
    if tqdm is not None:
        tqdm.write("")
    print(json.dumps(payload, indent=2))


# ----------------------------- #
# Feature importance helper
# ----------------------------- #
def _try_feature_importances(pipe, pre, num_cols, cat_cols) -> Optional[pd.DataFrame]:
    """
    Group OneHot columns back to source categorical feature names when possible.
    Returns DataFrame with ['group','importance'] or None.
    """
    try:
        mdl = pipe.named_steps.get("mdl", None)
        if mdl is None or not hasattr(mdl, "feature_importances_"):
            return None

        pre_ct = pipe.named_steps.get("pre")
        if pre_ct is None:
            return None

        # Find categorical pipeline to get expanded names
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

        feat_names = list(num_cols) + cat_feature_names
        importances = mdl.feature_importances_
        if len(importances) != len(feat_names):
            return pd.DataFrame(
                {"group": [f"f{i}" for i in range(len(importances))], "importance": importances}
            )

        fi = pd.DataFrame({"feature": feat_names, "importance": importances})

        def group_name(s: str) -> str:
            for c in cat_cols:
                if s.startswith(c + "_"):
                    return c
            return s

        fi["group"] = fi["feature"].map(group_name)
        fi_grouped = (
            fi.groupby("group", as_index=False)["importance"]
            .sum()
            .sort_values("importance", ascending=False)
        )
        return fi_grouped[["group", "importance"]]
    except Exception:
        return None


if __name__ == "__main__":
    run()
