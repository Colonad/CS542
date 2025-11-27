# src/realestate/targets.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

__all__ = [
    "TargetTransform",
    "RegressionMetrics",
    "compute_dollar_metrics",
]


@dataclass
class TargetTransform:
    """
    Handle target transformations for regression models.

    Designed for house price prediction where the raw target is a dollar
    price. Supports:

      • Training in log space via log1p(price).
      • Optional Duan smearing correction when mapping predictions back to
        dollar space.
      • Safe, typed helpers to go from DataFrame → numpy arrays and back.

    Typical usage inside train.py / train_sweep.py
    ----------------------------------------------
    >>> tt = TargetTransform(name="price", log_transform=True, duan_smearing=True)
    >>> y_train = tt.extract(train_df)               # Series of prices ($)
    >>> y_test = tt.extract(test_df)
    >>> y_train_fit = tt.transform_for_fit(y_train)  # log1p(price) array
    >>> pipe.fit(X_train, y_train_fit)
    >>> yhat_train_fit = pipe.predict(X_train)
    >>> tt.fit_smearing(y_train_fit, yhat_train_fit)
    >>> yhat_test_fit = pipe.predict(X_test)
    >>> yhat_test = tt.inverse(yhat_test_fit)        # back to dollars
    """

    # Column name of the raw target in the DataFrame.
    name: str = "price"

    # Whether to use log1p(price) for training.
    log_transform: bool = True

    # Whether to apply Duan's smearing correction when inverting.
    duan_smearing: bool = True

    # Learned at runtime after fitting (None if not used).
    smear_factor_: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Extraction
    # ------------------------------------------------------------------ #
    def extract(self, df: pd.DataFrame) -> pd.Series:
        """
        Extract the target column as a Series.

        Raises
        ------
        KeyError
            If the configured target column is not present.
        """
        if self.name not in df.columns:
            raise KeyError(f"Target column '{self.name}' not found in DataFrame.")
        # Copy to avoid chained assignment surprises.
        return df[self.name].copy()

    # ------------------------------------------------------------------ #
    # Forward transform: dollars → training space
    # ------------------------------------------------------------------ #
    def transform_for_fit(self, y: pd.Series) -> np.ndarray:
        """
        Transform raw dollar targets into the space used for model fitting.

        If `log_transform=True`, this returns log1p(y) as float64.
        Otherwise, it returns y as float64 with no transformation.

        Notes
        -----
        • Negative targets are rejected if log_transform=True.
        • Missing values should already have been dropped in IO; if they
          remain here they propagate as NaNs into the model.
        """
        arr = y.to_numpy(dtype=float)
        if self.log_transform:
            if np.any(arr < 0):
                raise ValueError(
                    "Negative target values encountered while log_transform=True. "
                    "Check data cleaning or disable log_transform in config."
                )
            return np.log1p(arr)
        return arr

    # ------------------------------------------------------------------ #
    # Duan smearing factor
    # ------------------------------------------------------------------ #
    def fit_smearing(self, y_fit_true: np.ndarray, y_fit_pred: np.ndarray) -> None:
        """
        Estimate Duan's smearing factor when training in log space.

        Let
            y_fit_true = log1p(price_true)
            y_fit_pred = model(X) in log space

        Define residuals in transformed space:
            eps = y_fit_true - y_fit_pred

        Duan's smearing factor is:
            smear = E[exp(eps)]

        When inverting a prediction y_hat in log space, the original-dollar
        scale estimate is:

            price_hat ≈ expm1(y_hat) * smear

        (When smear=1.0 this reduces to the usual inverse log1p.)
        If `log_transform=False` or `duan_smearing=False`, this sets
        `smear_factor_ = None`.
        """
        if not self.log_transform or not self.duan_smearing:
            self.smear_factor_ = None
            return

        if y_fit_true.shape != y_fit_pred.shape:
            raise ValueError(
                f"Shape mismatch when computing Duan smearing: "
                f"true={y_fit_true.shape}, pred={y_fit_pred.shape}"
            )

        eps = (y_fit_true - y_fit_pred).astype(float)
        if eps.size == 0:
            # Degenerate case; fall back to neutral correction.
            self.smear_factor_ = 1.0
        else:
            self.smear_factor_ = float(np.mean(np.exp(eps)))

    # ------------------------------------------------------------------ #
    # Inverse transform: model preds → dollars
    # ------------------------------------------------------------------ #
    def inverse(self, y_pred_fit: np.ndarray) -> np.ndarray:
        """
        Map predictions from fitting space back to dollar space.

        Parameters
        ----------
        y_pred_fit : np.ndarray
            Predictions in the same space that was used for model fitting:
              • If log_transform=True: log1p(price_hat)
              • Else: price_hat directly (already dollars).

        Returns
        -------
        np.ndarray
            Predictions in dollars, suitable for MAE/RMSE/R² against the
            original target.

        Behavior
        --------
        • If log_transform=False → no-op (cast to float64).
        • If log_transform=True:
              out = expm1(y_pred_fit)
              if Duan smearing is fitted:
                  out *= smear_factor_
        """
        y_pred_fit = np.asarray(y_pred_fit, dtype=float)

        if not self.log_transform:
            return y_pred_fit

        out = np.expm1(y_pred_fit)
        smear = self.smear_factor_
        if self.duan_smearing and smear is not None and np.isfinite(smear):
            out = out * smear
        return out

    # ------------------------------------------------------------------ #
    # Serialization for metrics JSON
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the target-transform configuration and learned state.
        Useful to embed in metrics JSON.
        """
        return {
            "name": self.name,
            "log_transform": self.log_transform,
            "duan_smearing": bool(self.duan_smearing),
            "smear_factor": float(self.smear_factor_) if self.smear_factor_ is not None else None,
        }


@dataclass
class RegressionMetrics:
    """
    Container for standard regression metrics in dollar space.
    """
    mae: float
    rmse: float
    r2: float

    @classmethod
    def from_arrays(cls, y_true: np.ndarray, y_pred: np.ndarray) -> "RegressionMetrics":
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return cls(
            mae=float(mean_absolute_error(y_true, y_pred)),
            rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
            r2=float(r2_score(y_true, y_pred)),
        )

    def as_dict(self) -> Dict[str, float]:
        return {"MAE": self.mae, "RMSE": self.rmse, "R2": self.r2}


def compute_dollar_metrics(
    y_true: np.ndarray,
    y_baseline: np.ndarray,
    y_model: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Compute MAE/RMSE/R² for both a baseline and a model in *dollar space*.

    Parameters
    ----------
    y_true : array-like
        Ground-truth prices in dollars.
    y_baseline : array-like
        Baseline predictions in dollars (e.g., zip×year median).
    y_model : array-like
        Model predictions in dollars, typically obtained by applying
        `TargetTransform.inverse` to model outputs.

    Returns
    -------
    dict
        {
          "baseline": {"MAE": ..., "RMSE": ..., "R2": ...},
          "model":    {"MAE": ..., "RMSE": ..., "R2": ...},
        }
    """
    base = RegressionMetrics.from_arrays(y_true, y_baseline)
    mdl = RegressionMetrics.from_arrays(y_true, y_model)
    return {"baseline": base.as_dict(), "model": mdl.as_dict()}
