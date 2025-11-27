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

def _usd(x: float | int | None) -> str:
    try:
        return f"${float(x):,.0f}"
    except Exception:
        return "—"
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

        # Be robust to potential naming differences ('n' vs 'count', 'mae' vs 'MAE')
        n_col = "n" if "n" in agg.columns else ("count" if "count" in agg.columns else None)
        mae_col = "mae" if "mae" in agg.columns else ("MAE" if "MAE" in agg.columns else None)
        if n_col is None or mae_col is None:
            raise KeyError(f"error_slices: expected count/MAE columns in agg, got {list(agg.columns)}")

        # Build label → mae map; iterate rows as namedtuples (single value from itertuples)
        results[g] = {
            f"{row.Index} (n={int(getattr(row, n_col))})": float(getattr(row, mae_col))
            for row in agg.itertuples(index=True)
        }

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
    metrics: Optional[Mapping[str, Any]] = None,
    run_id: Optional[str] = None,
    model_kind: str = "model",
) -> None:
    """
    Save quick diagnostic plots with informative titles/labels.

    If 'metrics' is provided, the titles include MAE/RMSE/R².
    """
    plots_cfg = cfg_eval.get("plots", {}) if isinstance(cfg_eval, Mapping) else {}
    if not any(plots_cfg.get(k, False) for k in ("pred_vs_actual", "residuals_vs_price")):
        return

    figures_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Force headless backend to avoid Wayland/Qt warnings
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: E402
        from matplotlib.ticker import FuncFormatter  # noqa: E402
    except Exception:
        # matplotlib not available; silently skip
        return

    # Pull nice numbers for titles if available
    m = metrics.get("model", {}) if isinstance(metrics, Mapping) else {}
    mae = m.get("MAE")
    rmse = m.get("RMSE")
    r2 = m.get("R2")

    title_suffix = []
    if run_id:
        title_suffix.append(f"run {run_id}")
    if model_kind:
        title_suffix.append(model_kind)
    # e.g., "run 2025-11-27_002753 · xgboost"
    suffix_str = " · ".join(title_suffix) if title_suffix else None

    # Tick formatter: dollars
    fmt_usd = FuncFormatter(lambda x, pos: _usd(x))

    # Predicted vs Actual scatter
    if plots_cfg.get("pred_vs_actual", False):
        fig = plt.figure()
        ax = fig.gca()
        ax.scatter(df[actual_col], df[pred_col], s=10, alpha=0.35)
        ax.set_xlabel("Actual Sale Price (USD)")
        ax.set_ylabel("Predicted Sale Price (USD)")
        ax.xaxis.set_major_formatter(fmt_usd)
        ax.yaxis.set_major_formatter(fmt_usd)

        # y=x reference
        try:
            lo = float(np.nanmin([df[actual_col].min(), df[pred_col].min()]))
            hi = float(np.nanmax([df[actual_col].max(), df[pred_col].max()]))
            ax.plot([lo, hi], [lo, hi], linewidth=1)
        except Exception:
            pass

        main = "Predicted vs Actual — Test Set"
        if mae is not None and rmse is not None and r2 is not None:
            main += f"  |  MAE={_usd(mae)}, RMSE={_usd(rmse)}, R²={r2:.3f}"
        if suffix_str:
            main += f"  ({suffix_str})"
        ax.set_title(main)
        fig.tight_layout()
        fig.savefig(figures_dir / "pred_vs_actual.png", dpi=120)
        plt.close(fig)

    # Residuals vs Actual price
    if plots_cfg.get("residuals_vs_price", False):
        fig = plt.figure()
        ax = fig.gca()
        resid = df[pred_col].to_numpy(dtype=float) - df[actual_col].to_numpy(dtype=float)
        ax.scatter(df[actual_col], resid, s=10, alpha=0.35)
        ax.axhline(0.0, linewidth=1)
        ax.set_xlabel("Actual Sale Price (USD)")
        ax.set_ylabel("Residual (Predicted − Actual, USD)")
        ax.xaxis.set_major_formatter(fmt_usd)
        ax.yaxis.set_major_formatter(fmt_usd)

        main = "Residuals vs Actual Price — Test Set"
        if mae is not None and rmse is not None and r2 is not None:
            main += f"  |  MAE={_usd(mae)}, RMSE={_usd(rmse)}, R²={r2:.3f}"
        if suffix_str:
            main += f"  ({suffix_str})"
        ax.set_title(main)
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
    Saves calibration_bins.csv and calibration_curve.png, and returns summary.
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

    # Plot calibration curve with informative labels
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: E402
        from matplotlib.ticker import FuncFormatter  # noqa: E402

        fig = plt.figure()
        ax = fig.gca()
        ax.scatter(grp["mean_pred"], grp["mean_actual"], s=22, alpha=0.7)

        lo = float(np.nanmin([grp["mean_pred"].min(), grp["mean_actual"].min()]))
        hi = float(np.nanmax([grp["mean_pred"].max(), grp["mean_actual"].max()]))
        ax.plot([lo, hi], [lo, hi], linewidth=1)

        fmt_usd = FuncFormatter(lambda x, pos: _usd(x))
        ax.xaxis.set_major_formatter(fmt_usd)
        ax.yaxis.set_major_formatter(fmt_usd)
        ax.set_xlabel("Mean Predicted Price per Bin (USD)")
        ax.set_ylabel("Mean Actual Price per Bin (USD)")
        ax.set_title(f"Calibration — {len(grp)} Bins  |  ECE_abs={_usd(ece_abs)}, ECE_rel={ece_rel:.2%}")

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
    cfg_path = Path(cfg_path).resolve()
    cfg = _load_yaml(cfg_path)

    # Project root = parent of 'configs/' if config lives there; else the config's folder
    cfg_dir = cfg_path.parent
    project_root = cfg_dir.parent if cfg_dir.name.lower() in {"configs", "config", "conf"} else cfg_dir

    # Resolve out_dir relative to the project root (stable regardless of CWD)
    out_dir_cfg = _get(cfg, ("paths", "out_dir"), "outputs")
    out_dir = Path(out_dir_cfg)
    if not out_dir.is_absolute():
        out_dir = (project_root / out_dir).resolve()

    metrics_dir = out_dir / "metrics"
    figures_dir = out_dir / "figures"
    preds_csv = out_dir / "preds" / "test_preds.csv"

    # Ensure dirs exist
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Optional: echo where we're writing (helps debugging)
    if tqdm is not None:
        tqdm.write(f"[Eval] writing outputs to: {out_dir}")


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

        # Load metrics.json (for annotating plot titles)
        metrics_payload = None
        metrics_path = metrics_dir / "metrics.json"
        if metrics_path.exists():
            try:
                metrics_payload = json.loads(metrics_path.read_text())
            except Exception:
                metrics_payload = None

        run_id = None
        model_kind = "model"
        if isinstance(metrics_payload, dict):
            run_id = (metrics_payload.get("run_info") or {}).get("run_id")
            model_kind = (metrics_payload.get("model") or {}).get("kind", model_kind)



    if want_pred_vs_actual or want_resid:
        _maybe_make_plots(
            df,
            figures_dir=figures_dir,
            actual_col=_get(cfg, ("target", "name"), "price"),
            pred_col="pred_model",
            cfg_eval=eval_cfg,
            metrics=metrics_payload,
            run_id=run_id,
            model_kind=model_kind,
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
