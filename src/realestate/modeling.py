# src/realestate/modeling.py
from __future__ import annotations

from typing import Any, Mapping, Optional
import os
import subprocess

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
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
def preprocessor(
    num_cols: list[str] | tuple[str, ...],
    cat_cols: list[str] | tuple[str, ...],
    *,
    cfg: Optional[Mapping[str, Any]] = None,
) -> ColumnTransformer:
    """
    Build a leakage-safe preprocessing transformer:

      - Numeric: SimpleImputer(strategy="median"|...)
      - Categorical: SimpleImputer(strategy="most_frequent"|...) -> OneHotEncoder

    Honors (if present in cfg['preprocess']):
      - numeric_imputer: "median" | "mean" | "constant"
      - categorical_imputer: "most_frequent" | "constant"
      - one_hot.handle_unknown: "ignore" (default) | "infrequent_if_exist" (sklearn>=1.4)
      - one_hot.min_frequency: int or float
    """
    p = (cfg or {}).get("preprocess", {}) if isinstance(cfg, Mapping) else {}

    # Numeric imputer
    num_strategy = p.get("numeric_imputer", "median")
    if num_strategy not in {"median", "mean", "constant"}:
        num_strategy = "median"
    num_imputer = SimpleImputer(strategy=num_strategy)

    # Categorical imputer
    cat_strategy = p.get("categorical_imputer", "most_frequent")
    if cat_strategy not in {"most_frequent", "constant"}:
        cat_strategy = "most_frequent"
    cat_imputer = SimpleImputer(strategy=cat_strategy)

    # One-hot encoder params
    oh_cfg = p.get("one_hot", {}) if isinstance(p, Mapping) else {}
    handle_unknown = oh_cfg.get("handle_unknown", "ignore")
    min_frequency = oh_cfg.get("min_frequency", 50)

    num_pipe = Pipeline([("imp", num_imputer)])
    cat_pipe = Pipeline(
        [
            ("imp", cat_imputer),
            ("oh", OneHotEncoder(handle_unknown=handle_unknown, min_frequency=min_frequency)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, list(num_cols)),
            ("cat", cat_pipe, list(cat_cols)),
        ],
        remainder="drop",
        sparse_threshold=0.3,  # allow sparse output for large OHEs
        n_jobs=None,
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
    Create an sklearn Pipeline:  preprocessor -> estimator

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
            # For explicit kinds we pull from that subsection; for "auto" we'll pick later
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
        params = _subset(model_cfg, {"alpha", "fit_intercept", "copy_X", "positive"})
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
        # Allow config to provide defaults under model.xgboost
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
            # predictor: let xgboost decide (often 'auto' -> 'cpu_predictor')

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
