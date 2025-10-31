# src/realestate/train.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional
from datetime import datetime
import hashlib

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump as joblib_dump

from . import io as io_mod
from .split import temporal_split
from .features import basic_time_feats, years_since_prev, apply_engineered, maybe_add_neighbors_via_cfg
from .baselines import median_by_zip_year
from .modeling import preprocessor, make_model

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

    # Load data (also writes schema snapshot)
    df = io_mod.load_data(data_csv, cfg=cfg)
    p.step("Loaded CSV + schema snapshot")

    # Row-wise features
    df = basic_time_feats(df)
    df = years_since_prev(df)
    df = apply_engineered(df, cfg)
    p.step("Derived features (time + engineered)")

    # Feature lists
    all_cols = set(df.columns)
    feats_cfg = cfg.get("features", {})
    num_cols = [c for c in feats_cfg.get("numeric", []) if c in all_cols]
    cat_cols = [c for c in feats_cfg.get("categorical", []) if c in all_cols]
    if not (num_cols or cat_cols):
        raise ValueError("No usable features found. Check configs.features.{numeric,categorical}.")
    p.step(f"Feature columns: num={len(num_cols)} cat={len(cat_cols)}")

    # Split
    cutoff = cfg.get("split", {}).get("cutoff", "2023-01-01")
    train_df, test_df = temporal_split(df, cutoff, date_col="sold_date")
    p.step("Temporal split complete")

    # Target config (needed for neighbor feature builder)
    target_col = cfg.get("target", {}).get("name", "price")
    use_log = bool(cfg.get("target", {}).get("log_transform", True))

    # Leakage-safe neighbors (train-only fit → map to test)
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
    y_train = train_df[target_col].copy()
    y_test = test_df[target_col].copy()
    y_train_fit = np.log1p(y_train.to_numpy(dtype=float)) if use_log else y_train.to_numpy(dtype=float)

    # Baseline
    base_pred = median_by_zip_year(train_df, test_df, target=target_col)
    p.step("Baseline computed (zip×year median)")

    # Model
    pre = preprocessor(num_cols, cat_cols, cfg=cfg)
    kind = cfg.get("model", {}).get("kind", "random_forest")
    pipe = make_model(kind, pre, cfg=cfg)
    p.step(f"Model created: {kind}")

    X_train = train_df[num_cols + cat_cols]
    X_test = test_df[num_cols + cat_cols]

    pipe.fit(X_train, y_train_fit)
    p.step("Model fit")

    pred_fit = pipe.predict(X_test)
    pred_model = np.expm1(pred_fit) if use_log else pred_fit
    p.step("Predictions generated")

    # Metrics
    metrics = _compute_metrics(y_test.to_numpy(dtype=float), base_pred.to_numpy(dtype=float), pred_model)
    p.step("Metrics computed")

    # Metadata
    run_info = {
        "run_id": run_name,
        "random_state": cfg.get("run", {}).get("random_state", 0),
        "data_csv": str(data_csv),
        "data_hash": _sha256_file(data_csv),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "cutoff": str(pd_timestamp_str(cutoff)),
        "val_cutoff": cfg.get("split", {}).get("val_cutoff", None),
        "feature_counts": {"numeric": len(num_cols), "categorical": len(cat_cols)},
        "model_kind": kind,
        "log_target": use_log,
    }
    payload = {
        "run_info": run_info,
        "target": {"name": target_col, "log_transform": use_log},
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
    # Extract reasonable params for logging; avoid dumping massive structures
    try:
        mdl = pipe.named_steps.get("mdl", None)
        if mdl is None:
            return {}
        params = mdl.get_params(deep=False)
        # prune verbose fields
        for k in list(params.keys()):
            if isinstance(params[k], (list, tuple, dict)) and len(str(params[k])) > 200:
                params[k] = str(type(params[k]))
        return params
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
