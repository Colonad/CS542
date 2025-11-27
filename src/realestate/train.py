# src/realestate/train.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional
from datetime import datetime
import hashlib
from copy import deepcopy

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump as joblib_dump
from sklearn.model_selection import train_test_split

from . import io as io_mod
from .split import temporal_split
from .features import (
    basic_time_feats,
    years_since_prev,
    apply_engineered,
    maybe_add_neighbors_via_cfg,
)
from .baselines import median_by_zip_year
from .modeling import preprocessor, make_model, describe_estimator
from .targets import TargetTransform, compute_dollar_metrics

try:
    from tqdm.auto import tqdm
except Exception:  # fallback if tqdm is somehow unavailable
    tqdm = None


class Prog:
    """Tiny progress helper."""
    def __init__(self, enabled: bool, total: int, desc: str):
        self.enabled = enabled and (tqdm is not None)
        self.t = tqdm(total=total, desc=desc, leave=True) if self.enabled else None

    def step(self, msg: str):
        if self.t:
            # show the latest substep; keep it short so it fits in one line
            self.t.set_postfix_str(msg[:60], refresh=True)
            self.t.update(1)

    def close(self):
        if self.t:
            self.t.close()


# ----------------------------- #
# Local helpers
# ----------------------------- #
def _random_split(df: pd.DataFrame, *, test_size: float = 0.2, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simple non-temporal split used when 'sold_date' is unavailable."""
    idx = df.index.to_numpy()
    tr_idx, te_idx = train_test_split(idx, test_size=test_size, random_state=seed, shuffle=True)
    return df.loc[tr_idx].copy(), df.loc[te_idx].copy()


def _has_datetime_sold_date(df: pd.DataFrame) -> bool:
    return ("sold_date" in df.columns) and pd.api.types.is_datetime64_any_dtype(df["sold_date"])




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

    # tiny synthetic train
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
                if dev.startswith("cuda"):  # normalize cuda:0, cuda:1, ...
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
        # XGBoost ≥ 2.0: use device='cuda' + tree_method='hist'
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
        # XGBoost 1.x: try legacy gpu_hist
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
    """
    Inspect the fitted XGBoost booster to infer the runtime device.
    Returns one of {"cuda","gpu","cpu","unknown"} (cuda:* normalized to "cuda").
    """
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
# Entrypoint
# ----------------------------- #
def run(cfg_path: str = "configs/config.yaml") -> None:
    cfg = _load_yaml(cfg_path)  # raises if empty/invalid
    if "paths" not in cfg or "data_csv" not in cfg["paths"]:
        raise KeyError(f"'paths.data_csv' is missing in {cfg_path}. Add it to point at your CSV.")

    # Decide how many steps we’ll show based on settings actually used
    timestamped = bool(cfg.get("run", {}).get("timestamped_outputs", True))
    save_pipeline = bool(cfg.get("train", {}).get("save_pipeline", True))
    save_fi = bool(cfg.get("train", {}).get("save_feature_importances", True))
    plots_cfg = cfg.get("eval", {}).get("plots", {}) or {}

    # Base steps + optional ones
    total_steps = 0
    # dirs + run naming + data load + feats + split + baseline + pre + fit + predict + metrics + save metrics + save preds
    total_steps += 12
    if save_pipeline:
        total_steps += 1
    if save_fi:
        total_steps += 1
    if timestamped:
        total_steps += 3  # copy into run/{metrics,preds,models}
    if bool(cfg.get("run", {}).get("save_config_snapshot", True)) and timestamped:
        total_steps += 1

    p = Prog(enabled=True, total=total_steps, desc="Training")

    # Paths
    data_csv = Path(cfg["paths"]["data_csv"])
    out_dir = Path(cfg["paths"]["out_dir"])
    models_dir = out_dir / "models"
    metrics_dir = out_dir / "metrics"
    preds_dir = out_dir / "preds"
    figures_dir = out_dir / "figures"
    for q in (models_dir, metrics_dir, preds_dir, figures_dir):
        q.mkdir(parents=True, exist_ok=True)
    p.step("Prepared output dirs")

    # Run name / timestamped
    run_name = cfg.get("run", {}).get("run_name") or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = out_dir / "run" / run_name if timestamped else None
    if run_dir:
        (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
        (run_dir / "preds").mkdir(parents=True, exist_ok=True)
        (run_dir / "models").mkdir(parents=True, exist_ok=True)
        p.step(f"Run folder: {run_name}")
    else:
        p.step("Run folder disabled")

    # ----------------------------------------------
    # Load data (with graceful fallback if no dates)
    # ----------------------------------------------
    relaxed_required = False
    try:
        df = io_mod.load_data(data_csv, cfg=cfg)
        p.step("Loaded CSV + schema snapshot")
    except ValueError as e:
        msg = str(e).lower()
        if "missing required columns" in msg and "sold_date" in msg:
            cfg_relaxed = deepcopy(cfg)
            cfg_relaxed.setdefault("data", {})
            cfg_relaxed["data"]["required_columns"] = ["price"]
            df = io_mod.load_data(data_csv, cfg=cfg_relaxed)
            relaxed_required = True
            p.step("Loaded CSV (relaxed required_columns=['price']) + schema snapshot")
        else:
            raise

    # Do we have a real 'sold_date' column?
    has_sold_date = _has_datetime_sold_date(df)

    # Row-wise features
    if has_sold_date:
        df = basic_time_feats(df)
        df = years_since_prev(df)
        p.step("Derived features (time)")
    else:
        p.step("No 'sold_date' — skipping time features")
    df = apply_engineered(df, cfg)
    p.step("Applied engineered features")

    # Feature lists
    all_cols = set(df.columns)
    feats_cfg = cfg.get("features", {})
    num_cols = [c for c in feats_cfg.get("numeric", []) if c in all_cols]
    cat_cols = [c for c in feats_cfg.get("categorical", []) if c in all_cols]
    if not (num_cols or cat_cols):
        raise ValueError("No usable features found. Check configs.features.{numeric,categorical}.")
    p.step(f"Feature columns: num={len(num_cols)} cat={len(cat_cols)}")

    # Split
    if has_sold_date:
        cutoff = cfg.get("split", {}).get("cutoff", "2023-01-01")
        train_df, test_df = temporal_split(df, cutoff, date_col="sold_date")
        p.step("Temporal split complete")
        cutoff_str = str(pd_timestamp_str(cutoff))
    else:
        rf = (cfg.get("split", {}) or {}).get("random_fallback", {}) or {}
        ts = float(rf.get("test_size", 0.2))
        seed = int(cfg.get("run", {}).get("random_state", 0))
        train_df, test_df = _random_split(df, test_size=ts, seed=seed)
        p.step(f"Random split complete (no 'sold_date', test_size={ts})")
        cutoff_str = None

    # Target config (needed for neighbor feature builder and TargetTransform)
    target_cfg = cfg.get("target", {}) or {}
    target_col = target_cfg.get("name", "price")
    use_log = bool(target_cfg.get("log_transform", True))
    use_duan = bool(target_cfg.get("duan_smearing", True))

    # Centralized target handler: dollars ↔ log1p(dollars)
    tt = TargetTransform(
        name=target_col,
        log_transform=use_log,
        duan_smearing=use_duan,
    )

    # Leakage-safe neighbors (train-only fit → map to test) — only when dates exist
    added_feats: list[str] = []
    if has_sold_date:
        train_df, test_df, added_feats = maybe_add_neighbors_via_cfg(
            train_df, test_df, cfg, target=target_col
        )
        if added_feats:
            p.step(f"Added neighbor feats: {', '.join(sorted(set(added_feats)))}")

        # Add any neighbor features that landed as numeric columns
        for f in added_feats:
            if f in train_df.columns and f not in num_cols and pd.api.types.is_numeric_dtype(train_df[f]):
                num_cols.append(f)

    # Refilter features to those that:
    #  - exist in BOTH train and test
    #  - have at least one non-null in TRAIN (so imputers don't warn)
    num_cols = [
        c for c in num_cols
        if (c in train_df.columns and c in test_df.columns and train_df[c].notna().any())
    ]
    cat_cols = [
        c for c in cat_cols
        if (c in train_df.columns and c in test_df.columns and train_df[c].notna().any())
    ]

    if not (num_cols or cat_cols):
        raise ValueError(
            "After split, no usable features remain (mismatch or all-NaN in train). "
            "Consider disabling neighbors or adjusting configs.features."
        )
    p.step(f"Usable features post-split: num={len(num_cols)} cat={len(cat_cols)}")

    if target_col not in train_df.columns:
        raise KeyError(f"Target column '{target_col}' not found.")

    # Raw dollar targets
    y_train = tt.extract(train_df)   # Series of prices ($)
    y_test = tt.extract(test_df)     # Series of prices ($)

    # Transformed target for fitting (log1p(price) if log_transform=True)
    y_train_fit = tt.transform_for_fit(y_train)

    # Baseline
    base_pred = median_by_zip_year(train_df, test_df, target=target_col)
    p.step("Baseline computed (zip×year median)")

    # Model
    pre = preprocessor(num_cols, cat_cols, cfg=cfg)
    kind = cfg.get("model", {}).get("kind", "random_forest")

    # ---- Device & backend selection (GPU-aware for XGBoost 1.x/2.x)
    requested_device = str(cfg.get("run", {}).get("device", "auto")).lower()
    resolved_device = "cpu"
    auto_switched = False

    # The model kind may be overridden below if GPU is usable
    kind = cfg.get("model", {}).get("kind", "random_forest")

    gpu_ok, gpu_note, gpu_params = (False, "", {})
    if requested_device in ("gpu", "auto"):
        gpu_ok, gpu_note, gpu_params = _probe_xgb_gpu()

    if gpu_ok:
        resolved_device = "gpu" if requested_device == "gpu" else "gpu(auto)"
        if kind != "xgboost":
            kind = "xgboost"
            cfg["model"]["kind"] = "xgboost"
            auto_switched = True

    # Configure XGBoost params depending on GPU usability and version
    if kind == "xgboost":
        xgb_cfg = cfg.setdefault("model", {}).setdefault("xgboost", {})
        if gpu_ok:
            xgb_cfg.update(gpu_params)       # passes device='cuda'+'hist' for ≥2.0
        else:
            xgb_cfg.clear()                  # ensure no stale GPU keys remain
            xgb_cfg["tree_method"] = "hist"  # force CPU path explicitly

            # Avoid leaving stale GPU-only params around
            xgb_cfg.pop("predictor", None)
            xgb_cfg.pop("device", None)
            resolved_device = "cpu"

    # Announce plan before building the model
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

    # Build the (possibly updated) model now
    pipe = make_model(kind, pre, cfg=cfg)




    p.step(f"Model created: {describe_estimator(pipe)}")

    X_train = train_df[num_cols + cat_cols]
    X_test  = test_df[num_cols + cat_cols]

    # === Fit model (with runtime device introspection) ===

    # === Fit model (with runtime device introspection) ===
    try:
        pipe.fit(X_train, y_train_fit)
        p.step("Model fit")
    except Exception:
        tm = str(cfg.get("model", {}).get("xgboost", {}).get("tree_method", "")).lower()
        dev_flag = str(cfg.get("model", {}).get("xgboost", {}).get("device", "")).lower()
        if kind == "xgboost" and ("gpu" in tm or dev_flag in ("cuda", "gpu")):
            # Retry on CPU
            cfg["model"]["xgboost"]["tree_method"] = "hist"
            cfg["model"]["xgboost"].pop("predictor", None)
            cfg["model"]["xgboost"].pop("device", None)
            pipe = make_model(kind, pre, cfg=cfg)
            pipe.fit(X_train, y_train_fit)
            p.step("GPU unavailable → fell back to CPU")
        else:
            raise

    # Determine what was actually used at runtime (not just what we planned)
    if kind == "xgboost":
        rt_dev = _xgb_runtime_device(pipe)
        used_device = "gpu" if (rt_dev == "gpu" or rt_dev.startswith("cuda")) else "cpu"
    else:
        used_device = "cpu"


    device_msg = f"[Device] requested={requested_device}  planned={resolved_device}  used={used_device}"
    if tqdm is not None:
        tqdm.write("")                 # blank line for separation
        tqdm.write(device_msg)         # own line, independent of the progress bar
    else:
        print("\n" + device_msg, flush=True)


    # Calibrate target transform / Duan smearing (if enabled).
    # Use in-sample predictions in the same space used for fitting
    # (log1p(price) if log_transform=True).
    train_pred_fit = pipe.predict(X_train)
    tt.fit_smearing(y_fit_true=y_train_fit, y_fit_pred=train_pred_fit)

    # Predict on test and map back to dollar space via TargetTransform.
    pred_fit = pipe.predict(X_test)
    pred_model = tt.inverse(pred_fit)   # always dollars here
    p.step("Predictions generated")

    # Metrics
    # Metrics in dollar space (y_test and both predictions are in $).
    metrics = compute_dollar_metrics(
        y_true=y_test.to_numpy(dtype=float),
        y_baseline=base_pred.to_numpy(dtype=float),
        y_model=np.asarray(pred_model, dtype=float),
    )
    p.step("Metrics computed")

    # Metadata
    run_info = {
        "run_id": run_name,
        "random_state": cfg.get("run", {}).get("random_state", 0),
        "data_csv": str(data_csv),
        "data_hash": _sha256_file(data_csv),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "cutoff": cutoff_str,
        "val_cutoff": cfg.get("split", {}).get("val_cutoff", None),
        "feature_counts": {"numeric": len(num_cols), "categorical": len(cat_cols)},
        "model_kind": kind,
        "log_target": use_log,
        "required_columns_relaxed": relaxed_required,
        "device_requested": requested_device,
        "device_used": used_device,

    }
    payload = {
        "run_info": run_info,
        "target": tt.to_dict(),

        "baseline": metrics["baseline"],
        "model": {"kind": kind, "params": _safe_model_params(pipe), **metrics["model"]},

    }

    # Save metrics
    (metrics_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    p.step("Saved metrics.json")
    if run_dir:
        (run_dir / "metrics" / "metrics.json").write_text(json.dumps(payload, indent=2))
        p.step("Copied metrics to run/")

    # Save preds
    keep_cols = [c for c in ["sold_date", "city", "state", "zip_code", target_col] if c in test_df.columns]
    preds_df = test_df[keep_cols].copy()
    preds_df["pred_baseline"] = base_pred.values
    preds_df["pred_model"] = pred_model
    preds_path = preds_dir / "test_preds.csv"
    preds_df.to_csv(preds_path, index=False)
    p.step("Saved predictions CSV")
    if run_dir:
        preds_df.to_csv(run_dir / "preds" / "test_preds.csv", index=False)
        p.step("Copied preds to run/")

    # Save model
    if save_pipeline:
        model_path = models_dir / cfg.get("train", {}).get("model_filename", "price_model.pkl")
        joblib_dump(pipe, model_path)
        p.step("Saved pipeline")
        if run_dir:
            joblib_dump(pipe, run_dir / "models" / model_path.name)
            p.step("Copied model to run/")

    # Feature importances
    if save_fi:
        fi_df = _try_feature_importances(pipe, pre, num_cols, cat_cols, X_train)
        if fi_df is not None:
            fi_path = models_dir / "feature_importances.csv"
            fi_df.to_csv(fi_path, index=False)
            p.step("Saved feature importances")
            if run_dir:
                fi_df.to_csv(run_dir / "models" / fi_path.name, index=False)
                p.step("Copied importances to run/")
        else:
            p.step("No feature importances available")

    # Config snapshot
    if bool(cfg.get("run", {}).get("save_config_snapshot", True)) and run_dir:
        snap_path = run_dir / "metrics" / "config.snapshot.yaml"
        snap_path.write_text(Path(cfg_path).read_text())
        p.step("Saved config snapshot")

    p.close()
    print("\n" + device_msg, flush=True)
    print(json.dumps(payload, indent=2))


# ----------------------------- #
# Helpers
# ----------------------------- #
def _load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p.resolve()}")
    data = yaml.safe_load(p.read_text())
    if data is None or not isinstance(data, dict) or not data:
        raise ValueError(
            f"Config file is empty or invalid YAML: {p.resolve()}\n"
            "Please populate configs/config.yaml (see example below)."
        )
    return data


def _compute_metrics(y_true: np.ndarray, y_base: np.ndarray, y_model: np.ndarray) -> dict[str, dict[str, float]]:
    # Use np.sqrt(MSE) so it works on all sklearn versions
    def rmse(a, b):
        return float(np.sqrt(mean_squared_error(a, b)))
    def mae(a, b):
        return float(mean_absolute_error(a, b))
    def r2(a, b):
        return float(r2_score(a, b))

    out = {
        "baseline": {
            "MAE": mae(y_true, y_base),
            "RMSE": rmse(y_true, y_base),
            "R2": r2(y_true, y_base),
        },
        "model": {
            "MAE": mae(y_true, y_model),
            "RMSE": rmse(y_true, y_model),
            "R2": r2(y_true, y_model),
        },
    }
    return out


def _safe_model_params(pipe) -> dict[str, Any]:
    """
    Extract model params for logging, but make them JSON-safe and compact:
      - Convert NumPy / pandas scalars to native Python types.
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
            return bool(np.isnan(x))  # works for numpy + floats
        except Exception:
            return False

    def _is_inf(x) -> bool:
        try:
            return bool(np.isinf(x))
        except Exception:
            return False

    def _to_native_scalar(x):
        # NumPy scalars -> Python
        if isinstance(x, np.generic):
            return x.item()
        # pandas Timestamp/Timedelta
        if "pandas" in str(type(x)):
            try:
                import pandas as pd  # already imported above; keep local to be safe
                if isinstance(x, pd.Timestamp):
                    return x.isoformat()
                if isinstance(x, pd.Timedelta):
                    return str(x)
            except Exception:
                pass
        # Plain scalars
        return x

    def _summarize_obj(x, depth: int = 0):
        # Normalize scalars early
        x = _to_native_scalar(x)

        # None/bool/int/float/str are fine after cleaning
        if x is None or isinstance(x, (bool, int)):
            return x
        if isinstance(x, float):
            if _is_nan(x) or _is_inf(x):
                return None
            return float(x)
        if isinstance(x, str):
            return x if len(x) <= MAX_STR else (x[:MAX_STR] + "…")

        # NumPy arrays
        if isinstance(x, np.ndarray):
            try:
                return f"ndarray(shape={x.shape}, dtype={x.dtype})"
            except Exception:
                return "ndarray"

        # pandas containers
        try:
            import pandas as pd  # safe import
            if isinstance(x, (pd.Series, pd.Index)):
                n = len(x)
                if n <= MAX_SEQ_ITEMS:
                    return [_summarize_obj(v, depth + 1) for v in x.tolist()]
                return f"{type(x).__name__}(len={n}, dtype={getattr(x, 'dtype', 'unknown')})"
            if isinstance(x, pd.DataFrame):
                return f"DataFrame(shape={x.shape}, columns={list(x.columns)[:8]}{'…' if x.shape[1] > 8 else ''})"
        except Exception:
            pass

        # Sequences (list/tuple/set)
        if isinstance(x, (list, tuple, set)):
            seq = list(x)
            n = len(seq)
            if n <= MAX_SEQ_ITEMS:
                return [_summarize_obj(v, depth + 1) for v in seq]
            return f"{type(x).__name__}(len={n})"

        # Dict-like
        if isinstance(x, dict):
            items = list(x.items())
            n = len(items)
            if n <= MAX_DICT_ITEMS:
                out = {}
                for k, v in items:
                    # keys should be strings for JSON
                    ks = str(k)
                    out[ks] = _summarize_obj(v, depth + 1)
                return out
            # Too big — summarize
            try:
                preview_keys = [str(k) for k, _ in items[:8]]
            except Exception:
                preview_keys = []
            return {"__summary__": f"dict(len={n})", "__preview_keys__": preview_keys}

        # Callables / modules / classes — summarize by name
        if callable(x):
            try:
                return f"<callable {getattr(x, '__name__', type(x).__name__)}>"
            except Exception:
                return "<callable>"
        if hasattr(x, "__module__") and hasattr(x, "__class__"):
            # Try a short, safe repr; fall back to type name
            try:
                r = repr(x)
                if len(r) <= MAX_STR:
                    return r
                return f"<{x.__class__.__name__}>"
            except Exception:
                return f"<{x.__class__.__name__}>"

        # Fallback
        try:
            r = repr(x)
            return r if len(r) <= MAX_STR else (r[:MAX_STR] + "…")
        except Exception:
            return f"<unrepr {type(x).__name__}>"

    try:
        mdl = pipe.named_steps.get("mdl", None)
        if mdl is None:
            return {}

        # Shallow params to avoid exploding into nested estimators
        raw = mdl.get_params(deep=False)

        # Clean each param
        cleaned = {}
        for k, v in raw.items():
            cv = _summarize_obj(v, 0)
            # Final guard: ensure JSON-safe float values
            if isinstance(cv, float) and (_is_nan(cv) or _is_inf(cv)):
                cv = None
            cleaned[str(k)] = cv

        # Optionally drop known noisy keys if present (kept conservative)
        for noisy in ("feature_weights", "feature_types"):
            if noisy in cleaned and not isinstance(cleaned[noisy], (str, int, float, bool, type(None))):
                cleaned[noisy] = str(type(cleaned[noisy]).__name__)

        # Return with keys sorted for stable diffs
        return {k: cleaned[k] for k in sorted(cleaned)}
    except Exception:
        return {}



def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return f"sha256:{h.hexdigest()}"


def _try_feature_importances(pipe, pre, num_cols, cat_cols, X_train: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Try to export grouped feature importances for tree models.
    Returns a DataFrame with columns: ['feature','importance'] or None.
    """
    try:
        mdl = pipe.named_steps.get("mdl", None)
        if mdl is None or not hasattr(mdl, "feature_importances_"):
            return None

        # Fit-transformer already fitted inside pipeline; reuse to get feature names
        # Rebuild encoder refs
        pre_ct = pipe.named_steps.get("pre")
        if pre_ct is None:
            return None

        # Fetch OHE to get expanded cat names
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
            # Fallback: return raw importances with generic indices
            return pd.DataFrame({"feature": [f"f{i}" for i in range(len(importances))],
                                 "importance": importances})

        fi = pd.DataFrame({"feature": feat_names, "importance": importances})

        # Group OHE back to original category columns by prefix "col_"
        def group_name(s: str) -> str:
            for c in cat_cols:
                if s.startswith(c + "_"):
                    return c
            return s

        cat_cols = cat_cols_used  # ensure we group by the ones actually used
        fi["group"] = fi["feature"].map(group_name)
        fi_grouped = fi.groupby("group", as_index=False)["importance"].sum().sort_values(
            "importance", ascending=False
        )
        return fi_grouped
    except Exception:
        return None


def pd_timestamp_str(x: Any) -> str:
    try:
        return pd.Timestamp(x).isoformat()
    except Exception:
        return str(x)


if __name__ == "__main__":
    run()
