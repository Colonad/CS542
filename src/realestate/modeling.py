# src/realestate/modeling.py
from __future__ import annotations

from typing import Any, Mapping, Optional

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

__all__ = ["preprocessor", "make_model"]

# Try to import XGBoost (optional)
_HAS_XGB = False
try:
    from xgboost import XGBRegressor  # type: ignore
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


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

      - Numeric: SimpleImputer(strategy="median")
      - Categorical: SimpleImputer(strategy="most_frequent")
                     -> OneHotEncoder(handle_unknown="ignore", min_frequency=50)

    Notes
    -----
    - If `cfg['preprocess']` is provided, we honor:
        preprocess.numeric_imputer        ("median" | "mean" | "constant")
        preprocess.categorical_imputer    ("most_frequent" | "constant")
        preprocess.one_hot.min_frequency  (int | float)
        preprocess.one_hot.handle_unknown ("ignore" | "infrequent_if_exist")  # sklearn>=1.4
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

    # Build sub-pipelines
    num_pipe = Pipeline(
        steps=[
            ("imp", num_imputer),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
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
        sparse_threshold=0.3,  # let OHE keep sparse where helpful
        n_jobs=None,
    )


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
def make_model(
    kind: str,
    pre: ColumnTransformer,
    cfg: Optional[Mapping[str, Any]] = None,
) -> Pipeline:
    """
    Create an sklearn Pipeline composed of:
        preprocessor -> estimator

    Supported `kind`:
      - "ridge"
      - "random_forest"
      - "xgboost" (only if xgboost is installed)

    Parameters
    ----------
    kind : str
        Estimator kind.
    pre : ColumnTransformer
        Output of `preprocessor(...)`.
    cfg : Mapping, optional
        Full config dictionary; we read hyperparameters under `cfg["model"][<kind>]`.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    model_cfg = {}
    if isinstance(cfg, Mapping):
        model_section = cfg.get("model", {})
        if isinstance(model_section, Mapping):
            model_cfg = model_section.get(kind, {}) or {}

    kind = kind.lower().strip()

    if kind == "ridge":
        # Ridge does not accept random_state; we pass only known params
        params = _subset(model_cfg, {"alpha", "fit_intercept", "copy_X", "positive"})
        est = Ridge(**params)

    elif kind == "random_forest":
        # Pass through common RF params; sklearn ignores unknowns with TypeError, so subset to be safe
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

    elif kind == "xgboost":
        if not _HAS_XGB:
            raise ImportError(
                "Requested kind='xgboost' but xgboost is not installed. "
                "Install it or choose a different model."
            )
        # Common, safe XGBRegressor params
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
            "n_jobs",
            "random_state",
        }
        params = _subset(model_cfg, allowed)
        # Keep objective consistent with log-price regression (squared error)
        if "objective" not in params:
            params["objective"] = "reg:squarederror"
        est = XGBRegressor(**params)

    else:
        raise ValueError(f"Unknown model kind: {kind!r}. Choose 'ridge', 'random_forest', or 'xgboost'.")

    return Pipeline([("pre", pre), ("mdl", est)])


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def _subset(d: Mapping[str, Any], keys: set[str]) -> dict[str, Any]:
    """Return a shallow copy of d with only items whose key is in keys."""
    return {k: v for k, v in d.items() if k in keys}
