# src/realestate/evaluate.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd
import yaml

__all__ = ["error_slices", "run"]

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


# --------------------------------------------------------------------------- #
# Progress helper
# --------------------------------------------------------------------------- #
class Prog:
    def __init__(self, enabled: bool, total: int, desc: str):
        self.enabled = enabled and (tqdm is not None)
        self.t = tqdm(total=total, desc=desc, leave=True) if self.enabled else None

    def step(self, msg: str):
        if self.t:
            self.t.set_postfix_str(msg[:60], refresh=True)
            self.t.update(1)

    def close(self):
        if self.t:
            self.t.close()


# --------------------------------------------------------------------------- #
# Quiet datetime + ZIP helpers
# --------------------------------------------------------------------------- #
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


def _normalize_zip_str(zser: pd.Series) -> pd.Series:
    """Normalize ZIP-like values to clean 5-char strings."""
    if pd.api.types.is_numeric_dtype(zser):
        z = pd.to_numeric(zser, errors="coerce").astype("Int64").astype(str)
    else:
        z = zser.astype(str)
    z = (
        z.str.strip()
         .str.replace("<NA>", "", regex=False)
         .str.replace(r"\.0$", "", regex=True)
         .str.replace(r"[^\d]", "", regex=True)
    )
    # Pad 1–5 digits to 5 (US-style)
    return z.where(~z.str.fullmatch(r"\d{1,5}"), z.str.zfill(5))


def _quiet_dates_in_df(df: pd.DataFrame) -> pd.DataFrame:
    """Quietly coerce any '...date...' columns or datetime-like columns."""
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if ("date" in c.lower()) or pd.api.types.is_datetime64_any_dtype(s):
            out[c] = _coerce_datetime_quiet(s)
    return out


# --------------------------------------------------------------------------- #
# Core slicing
# --------------------------------------------------------------------------- #
def error_slices(
    preds_csv: str | Path,
    *,
    group_cols: Iterable[str] = ("state", "zip_code"),
    actual_col: str = "price",
    pred_col: str = "pred_model",
    topk: int = 20,
    min_count: int = 10,
) -> Dict[str, Dict[str, float]]:
    """
    Compute mean absolute error by each group column, with a minimum group size filter.

    Returns
    -------
    dict
        { group_col: { "group_value (n=XX)": mae, ... }, ... } with at most `topk` entries per group_col.
    """
    preds_csv = Path(preds_csv)
    if not preds_csv.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {preds_csv.resolve()}")

    df = pd.read_csv(preds_csv, low_memory=False)
    # Quietly coerce any date-ish columns (harmless if none)
    df = _quiet_dates_in_df(df)

    if actual_col not in df.columns or pred_col not in df.columns:
        raise KeyError(
            f"Missing required columns '{actual_col}' and/or '{pred_col}' in {preds_csv}"
        )

    # Normalize grouping columns; special-case zip_code for numeric artifacts (e.g., '12345.0')
    for g in group_cols:
        if g in df.columns:
            if g == "zip_code":
                df[g] = _normalize_zip_str(df[g])
            else:
                df[g] = df[g].astype(str).str.strip()

    # Absolute error
    y = df[actual_col].to_numpy(dtype=float)
    yhat = df[pred_col].to_numpy(dtype=float)
    abs_err = np.abs(y - yhat)
    df["_abs_err_"] = abs_err

    results: Dict[str, Dict[str, float]] = {}
    for g in group_cols:
        if g not in df.columns:
            continue
        agg = (
            df.dropna(subset=[g])
              .groupby(g)
              .agg(mae=("_abs_err_", "mean"), n=(actual_col, "size"))
              .sort_values("mae", ascending=False)
        )
        # keep sufficiently large groups
        agg = agg[agg["n"] >= int(min_count)].head(int(topk))
        results[g] = {f"{idx} (n={int(row.n)})": float(row.mae) for idx, row in agg.itertuples()}

    if "_abs_err_" in df.columns:
        del df["_abs_err_"]
    return results


# --------------------------------------------------------------------------- #
# Optional quick plots (controlled by config)
# --------------------------------------------------------------------------- #
def _maybe_make_plots(
    df: pd.DataFrame,
    *,
    figures_dir: Path,
    actual_col: str,
    pred_col: str,
    cfg_eval: Mapping[str, Any],
) -> None:
    plots_cfg = cfg_eval.get("plots", {}) if isinstance(cfg_eval, Mapping) else {}
    if not any(plots_cfg.get(k, False) for k in ("pred_vs_actual", "residuals_vs_price")):
        return

    figures_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Force headless backend to avoid Wayland/Qt warnings
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: E402
    except Exception:
        # matplotlib not available; silently skip
        return

    # Predicted vs Actual scatter
    if plots_cfg.get("pred_vs_actual", False):
        fig = plt.figure()
        ax = fig.gca()
        ax.scatter(df[actual_col], df[pred_col], s=8, alpha=0.4)
        ax.set_xlabel("Actual price")
        ax.set_ylabel("Predicted price")
        ax.set_title("Predicted vs Actual")
        # y=x reference line (limits based on data range)
        try:
            lo = float(np.nanmin([df[actual_col].min(), df[pred_col].min()]))
            hi = float(np.nanmax([df[actual_col].max(), df[pred_col].max()]))
            ax.plot([lo, hi], [lo, hi], linewidth=1)
        except Exception:
            pass
        fig.tight_layout()
        fig.savefig(figures_dir / "pred_vs_actual.png", dpi=120)
        plt.close(fig)

    # Residuals vs Actual price
    if plots_cfg.get("residuals_vs_price", False):
        fig = plt.figure()
        ax = fig.gca()
        resid = df[pred_col].to_numpy(dtype=float) - df[actual_col].to_numpy(dtype=float)
        ax.scatter(df[actual_col], resid, s=8, alpha=0.4)
        ax.axhline(0.0, linewidth=1)
        ax.set_xlabel("Actual price")
        ax.set_ylabel("Residual (pred - actual)")
        ax.set_title("Residuals vs Actual price")
        fig.tight_layout()
        fig.savefig(figures_dir / "residuals_vs_price.png", dpi=120)
        plt.close(fig)


def _calibration(
    df: pd.DataFrame,
    *,
    actual_col: str,
    pred_col: str,
    bins: int,
    metrics_dir: Path,
    figures_dir: Path,
) -> dict:
    """
    Reliability diagram for regression (bin by predicted quantiles).
    Saves calibration_bins.csv and calibration_curve.png.
    """
    df = df[[actual_col, pred_col]].dropna().copy()
    if len(df) == 0:
        return {"bins": 0, "ECE_abs": None, "ECE_rel": None, "total_points": 0}

    # Use rank→qcut to avoid duplicate-edge issues on heavy ties
    r = df[pred_col].rank(method="first")
    try:
        df["_bin"] = pd.qcut(r, q=bins, labels=False, duplicates="drop")
    except Exception:
        # fallback: if qcut fails for extreme ties, force a single bin
        df["_bin"] = 0

    grp = (
        df.groupby("_bin")
          .agg(
              n=(actual_col, "size"),
              mean_pred=(pred_col, "mean"),
              mean_actual=(actual_col, "mean"),
              min_pred=(pred_col, "min"),
              max_pred=(pred_col, "max"),
          )
          .reset_index(drop=True)
    )
    if len(grp) == 0:
        return {"bins": 0, "ECE_abs": None, "ECE_rel": None, "total_points": 0}

    # ECE metrics
    diff = (grp["mean_pred"] - grp["mean_actual"]).abs()
    w = grp["n"] / grp["n"].sum()
    ece_abs = float(np.sum(w * diff))
    ece_rel = float(np.sum(w * (diff / np.maximum(grp["mean_actual"], 1e-9))))

    # Save CSV
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    grp.to_csv(metrics_dir / "calibration_bins.csv", index=False)

    # Plot calibration curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: E402
        fig = plt.figure()
        ax = fig.gca()
        ax.scatter(grp["mean_pred"], grp["mean_actual"], s=20, alpha=0.7)
        lo = float(np.nanmin([grp["mean_pred"].min(), grp["mean_actual"].min()]))
        hi = float(np.nanmax([grp["mean_pred"].max(), grp["mean_actual"].max()]))
        ax.plot([lo, hi], [lo, hi], linewidth=1)
        ax.set_xlabel("Mean predicted (per bin)")
        ax.set_ylabel("Mean actual (per bin)")
        ax.set_title("Calibration (Reliability) — Regression")
        fig.tight_layout()
        fig.savefig(figures_dir / "calibration_curve.png", dpi=120)
        plt.close(fig)
    except Exception:
        pass

    # Save a small JSON too
    calib_payload = {
        "bins": int(len(grp)),
        "ECE_abs": ece_abs,
        "ECE_rel": ece_rel,
        "total_points": int(df.shape[0]),
    }
    (metrics_dir / "calibration.json").write_text(json.dumps(calib_payload, indent=2))
    return calib_payload


# --------------------------------------------------------------------------- #
# CLI entrypoint
# --------------------------------------------------------------------------- #
def run(cfg_path: str = "configs/config.yaml") -> None:
    cfg = _load_yaml(cfg_path)

    out_dir = Path(_get(cfg, ("paths", "out_dir"), "outputs"))
    metrics_dir = out_dir / "metrics"
    figures_dir = out_dir / "figures"
    preds_csv = out_dir / "preds" / "test_preds.csv"

    metrics_dir.mkdir(parents=True, exist_ok=True)

    eval_cfg = _get(cfg, ("eval",), {}) or {}
    slice_cfg = _get(eval_cfg, ("slices",), {}) or {}
    group_cols = tuple(slice_cfg.get("group_cols", ["state", "zip_code"]))
    topk = int(slice_cfg.get("topk", 20))
    min_count = int(slice_cfg.get("min_count", 10))

    # Steps: slices, save json, optional plots, optional calibration
    want_pred_vs_actual = bool(eval_cfg.get("plots", {}).get("pred_vs_actual", False))
    want_resid = bool(eval_cfg.get("plots", {}).get("residuals_vs_price", False))
    want_calib = bool(eval_cfg.get("plots", {}).get("calibration_curve", False))
    total_steps = 3 + (1 if want_pred_vs_actual else 0) + (1 if want_resid else 0) + (1 if want_calib else 0)
    p = Prog(enabled=True, total=total_steps, desc="Evaluation")

    # Compute slices
    p.step("Reading predictions")
    slices = error_slices(
        preds_csv,
        group_cols=group_cols,
        actual_col=_get(cfg, ("target", "name"), "price"),
        pred_col="pred_model",
        topk=topk,
        min_count=min_count,
    )
    p.step("Computed error slices")

    # Save JSON
    (metrics_dir / "slices.json").write_text(json.dumps(slices, indent=2))
    p.step("Saved slices.json")

    # Optional quick plots
    if want_pred_vs_actual or want_resid or want_calib:
        df = pd.read_csv(preds_csv, low_memory=False)
        # Quiet date coercion + normalize ZIP if present to keep plots/calib stable
        df = _quiet_dates_in_df(df)
        if "zip_code" in df.columns:
            df["zip_code"] = _normalize_zip_str(df["zip_code"])

    if want_pred_vs_actual or want_resid:
        _maybe_make_plots(
            df,
            figures_dir=figures_dir,
            actual_col=_get(cfg, ("target", "name"), "price"),
            pred_col="pred_model",
            cfg_eval=eval_cfg,
        )
        p.step("Saved plots")

    # Calibration / reliability
    if want_calib:
        bins = int(eval_cfg.get("calibration", {}).get("bins", 20))
        calib = _calibration(
            df,
            actual_col=_get(cfg, ("target", "name"), "price"),
            pred_col="pred_model",
            bins=bins,
            metrics_dir=metrics_dir,
            figures_dir=figures_dir,
        )
        # If ECE metrics are None (empty), avoid formatting errors
        e_abs = "nan" if calib["ECE_abs"] is None else f"{calib['ECE_abs']:.1f}"
        e_rel = "nan" if calib["ECE_rel"] is None else f"{calib['ECE_rel']:.4f}"
        p.step(f"Saved calibration (ECE_abs={e_abs}, ECE_rel={e_rel})")

    p.close()
    print(json.dumps({"slices_written": str(metrics_dir / "slices.json"), "groups": list(slices.keys())}, indent=2))


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def _load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p.resolve()}")
    return yaml.safe_load(p.read_text())


def _get(cfg: Optional[Mapping[str, Any]], path: Iterable[str], default: Any = None) -> Any:
    cur: Any = cfg
    try:
        for key in path:
            if cur is None:
                return default
            cur = cur.get(key)
        return default if cur is None else cur
    except AttributeError:
        return default


if __name__ == "__main__":
    run()
