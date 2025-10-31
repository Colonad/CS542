# src/realestate/modeling.py
from __future__ import annotations

from typing import Any, Mapping, Optional
import os
import subprocess

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

__all__ = ["preprocessor", "make_model", "describe_estimator"]

# Optional XGBoost
_HAS_XGB = False
try:
    from xgboost import XGBRegressor  # type: ignore
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


# --------------------------------------------------------------------------- #
# GPU availability probe (works well on Colab/servers)
# --------------------------------------------------------------------------- #
def _gpu_available() -> bool:
    """
    Heuristic GPU check:
      1) CUDA_VISIBLE_DEVICES is set and not '-1'
      2) 'nvidia-smi -L' returns at least one GPU
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd and cvd != "-1":
        return True
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
            timeout=2,
        )
        return (out.returncode == 0) and ("GPU" in (out.stdout or ""))
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Preprocessing factory
# --------------------------------------------------------------------------- #
def _safe_one_hot_encoder(
    *,
    handle_unknown: str = "ignore",
    min_frequency: int | float | None = 50,
) -> OneHotEncoder:
    """
    Build a OneHotEncoder while being compatible with older scikit-learn versions.
    If the installed sklearn does not support `min_frequency`, we silently omit it.
    """
    try:
        # Newer sklearn (>=1.1) supports min_frequency
        return OneHotEncoder(handle_unknown=handle_unknown, min_frequency=min_frequency)
    except TypeError:
        # Fallback for older versions – still leakage-safe, just without frequency bucketing
        return OneHotEncoder(handle_unknown=handle_unknown)


def preprocessor(
    num_cols: list[str] | tuple[str, ...],
    cat_cols: list[str] | tuple[str, ...],
    *,
    cfg: Optional[Mapping[str, Any]] = None,
) -> ColumnTransformer:
    """
    Build a leakage-safe preprocessing transformer:

    Numeric:
      - SimpleImputer(strategy in {'median','mean','constant'})  # default: 'median'
      - Optional StandardScaler (if cfg['preprocess']['scaler'] == 'standard')

    Categorical:
      - SimpleImputer(strategy in {'most_frequent','constant'})  # default: 'most_frequent'
      - OneHotEncoder(handle_unknown='ignore', min_frequency=50 by default)
        (falls back to no min_frequency on old sklearns)

    Honors cfg['preprocess'] if present:
      preprocess:
        numeric_imputer: median | mean | constant
        categorical_imputer: most_frequent | constant
        one_hot:
          handle_unknown: ignore
          min_frequency: 50          # int or float in (0,1]
        scaler: null | standard
    """
    p = (cfg or {}).get("preprocess", {}) if isinstance(cfg, Mapping) else {}

    # ---------- Numeric pipeline ----------
    num_strategy = p.get("numeric_imputer", "median")
    if num_strategy not in {"median", "mean", "constant"}:
        num_strategy = "median"

    num_imputer_kwargs: dict[str, Any] = {"strategy": num_strategy}
    if num_strategy == "constant":
        # allow explicit fill_value override, else default 0.0 for numeric
        fv = p.get("numeric_fill_value", 0.0)
        num_imputer_kwargs["fill_value"] = fv
    num_imputer = SimpleImputer(**num_imputer_kwargs)

    scaler_name = (p.get("scaler") or "").strip().lower()
    if scaler_name == "standard":
        num_steps = [("imp", num_imputer), ("scaler", StandardScaler())]
    else:
        num_steps = [("imp", num_imputer)]

    # ---------- Categorical pipeline ----------
    cat_strategy = p.get("categorical_imputer", "most_frequent")
    if cat_strategy not in {"most_frequent", "constant"}:
        cat_strategy = "most_frequent"
    cat_imputer_kwargs: dict[str, Any] = {"strategy": cat_strategy}
    if cat_strategy == "constant":
        cat_imputer_kwargs["fill_value"] = p.get("categorical_fill_value", "missing")
    cat_imputer = SimpleImputer(**cat_imputer_kwargs)

    oh_cfg = p.get("one_hot", {}) if isinstance(p, Mapping) else {}
    handle_unknown = oh_cfg.get("handle_unknown", "ignore")
    min_frequency = oh_cfg.get("min_frequency", 50)

    oh = _safe_one_hot_encoder(handle_unknown=handle_unknown, min_frequency=min_frequency)

    cat_steps = [("imp", cat_imputer), ("oh", oh)]

    # ---------- ColumnTransformer ----------
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), list(num_cols)),
            ("cat", Pipeline(cat_steps), list(cat_cols)),
        ],
        remainder="drop",
        sparse_threshold=0.3,  # allow sparse output for large OHEs
        n_jobs=None,
        verbose=False,
    )


# --------------------------------------------------------------------------- #
# Model registry with auto device/model selection
# --------------------------------------------------------------------------- #
def make_model(
    kind: str,
    pre: ColumnTransformer,
    cfg: Optional[Mapping[str, Any]] = None,
) -> Pipeline:
    """
    Create an sklearn Pipeline: preprocessor -> estimator

    Supported `kind`:
      - "auto"          -> prefer XGBoost; GPU if available; else XGBoost CPU; else RandomForest
      - "xgboost"       -> XGBRegressor (gpu_hist if available, else hist)
      - "random_forest" -> RandomForestRegressor (CPU)
      - "ridge"         -> Ridge (CPU)
    """
    model_cfg: dict[str, Any] = {}
    if isinstance(cfg, Mapping):
        model_section = cfg.get("model", {})
        if isinstance(model_section, Mapping):
            model_cfg = model_section.get(kind, {}) or {}

    requested = kind.lower().strip()

    # Resolve 'auto' to a concrete estimator
    if requested == "auto":
        xgb_enabled = True
        if isinstance(cfg, Mapping):
            xgb_enabled = (cfg.get("model", {}).get("xgboost", {}).get("enabled", True) is not False)
        if _HAS_XGB and xgb_enabled:
            requested = "xgboost"
        else:
            requested = "random_forest"

    # --- Ridge ---
    if requested == "ridge":
        params = _subset(model_cfg, {"alpha", "fit_intercept", "copy_X", "positive", "random_state"})
        est = Ridge(**params)

    # --- Random Forest (CPU) ---
    elif requested == "random_forest":
        allowed = {
            "n_estimators",
            "criterion",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "min_weight_fraction_leaf",
            "max_features",
            "max_leaf_nodes",
            "min_impurity_decrease",
            "bootstrap",
            "oob_score",
            "n_jobs",
            "random_state",
            "verbose",
            "warm_start",
            "ccp_alpha",
            "max_samples",
        }
        params = _subset(model_cfg, allowed)
        est = RandomForestRegressor(**params)

    # --- XGBoost (GPU/CPU auto) ---
    elif requested == "xgboost":
        if not _HAS_XGB:
            raise ImportError(
                "Requested kind='xgboost' but xgboost is not installed. "
                "Install it (e.g., `pip install xgboost`) or choose a different model."
            )
        allowed = {
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "reg_lambda",
            "reg_alpha",
            "gamma",
            "min_child_weight",
            "tree_method",
            "predictor",
            "n_jobs",
            "random_state",
        }
        xgb_cfg = {}
        if isinstance(cfg, Mapping):
            xgb_cfg = cfg.get("model", {}).get("xgboost", {}) or {}
        params = _subset(xgb_cfg, allowed)

        # Auto-choose backend
        if _gpu_available():
            params.setdefault("tree_method", "gpu_hist")
            params.setdefault("predictor", "gpu_predictor")
        else:
            params.setdefault("tree_method", "hist")
            # predictor left default (xgboost will pick cpu_predictor)

        params.setdefault("objective", "reg:squarederror")
        est = XGBRegressor(**params)

    else:
        raise ValueError(
            f"Unknown model kind: {kind!r}. "
            "Choose 'auto', 'xgboost', 'random_forest', or 'ridge'."
        )

    return Pipeline([("pre", pre), ("mdl", est)])


# --------------------------------------------------------------------------- #
# Small utility to print the chosen backend in progress lines
# --------------------------------------------------------------------------- #
def describe_estimator(pipe: Pipeline) -> str:
    """
    Return a short description, e.g.:
      - 'xgboost [gpu_hist]'
      - 'xgboost [hist]'
      - 'randomforestregressor'
      - 'ridge'
    """
    try:
        mdl = pipe.named_steps.get("mdl", None)
        if mdl is None:
            return "unknown"
        name = mdl.__class__.__name__.lower()
        if name == "xgbregressor":
            # Try direct attr, then params
            tm = getattr(mdl, "tree_method", None)
            if not tm and hasattr(mdl, "get_xgb_params"):
                tm = mdl.get_xgb_params().get("tree_method")
            return f"xgboost [{tm}]" if tm else "xgboost"
        return name
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def _subset(d: Mapping[str, Any], keys: set[str]) -> dict[str, Any]:
    """Return a shallow copy of d with only items whose key is in keys."""
    return {k: v for k, v in d.items() if k in keys}
