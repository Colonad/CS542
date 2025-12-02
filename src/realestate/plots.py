#!/usr/bin/env python3
"""
realestate.plots
================

Figure generation for the CS542 Real Estate project.

Reads the best-model test predictions from:
    <out_dir>/preds/test_preds.csv

and generates:
    - pred_vs_actual_test.png         (scatter with y = x line)
    - resid_vs_price_test.png         (residual vs actual price)
    - calibration_test.png            (reliability diagram with ECE)

All figures are written to:
    <out_dir>/figures
where <out_dir> comes from configs/config.yaml.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import pandas as pd
import yaml


matplotlib.use("Agg")  # non-GUI backend for script usage

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _load_cfg(cfg_path: str | Path) -> dict:
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path.resolve()}")
    data = yaml.safe_load(cfg_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML in {cfg_path}")
    return data


def _paths_from_cfg(cfg: dict, preds_override: str | None = None, outdir_override: str | None = None):
    out_dir = Path(cfg["paths"]["out_dir"])
    preds_path = Path(preds_override) if preds_override else out_dir / "preds" / "test_preds.csv"
    fig_dir = Path(outdir_override) if outdir_override else out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Optional: grab run_id for titles
    metrics_path = out_dir / "metrics" / "phase6_sweep_metrics.json"
    run_id = None
    if metrics_path.exists():
        try:
            payload = json.loads(metrics_path.read_text())
            run_id = payload.get("run_info", {}).get("run_id")
        except Exception:
            run_id = None

    return preds_path, fig_dir, run_id


def _load_preds(preds_path: Path) -> pd.DataFrame:
    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {preds_path.resolve()}")

    df = pd.read_csv(preds_path, parse_dates=["sold_date"], infer_datetime_format=True)

    required = {"price", "pred_model"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{preds_path} must contain columns {sorted(required)}; "
            f"missing: {sorted(missing)}"
        )
    return df


def _currency_formatter():
    # Format ticks as $xxx,xxx
    return StrMethodFormatter("${x:,.0f}")


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    # vanilla R^2
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# ------------------------------------------------------------
# Plots
# ------------------------------------------------------------
def plot_pred_vs_actual(df: pd.DataFrame, out_path: Path, run_id: str | None = None):
    y_true = df["price"].to_numpy(dtype=float)
    y_pred = df["pred_model"].to_numpy(dtype=float)
    m = _compute_metrics(y_true, y_pred)

    lo = float(np.percentile(np.concatenate([y_true, y_pred]), 1))
    hi = float(np.percentile(np.concatenate([y_true, y_pred]), 99))
    lo = max(0.0, lo)
    hi = hi * 1.05

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.3)
    ax.plot([lo, hi], [lo, hi])

    ax.set_xlabel("Actual Sale Price (USD)")
    ax.set_ylabel("Predicted Sale Price (USD)")
    ax.xaxis.set_major_formatter(_currency_formatter())
    ax.yaxis.set_major_formatter(_currency_formatter())

    title = (
        f"Predicted vs Actual — Test Set  |  "
        f"MAE=${m['MAE']:,.0f}, RMSE={m['RMSE']:,.0f}, R²={m['R2']:.3f}"
    )
    if run_id:
        title += f"  (run {run_id})"
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plots] Saved {out_path}")


def plot_resid_vs_price(df: pd.DataFrame, out_path: Path, run_id: str | None = None):
    y_true = df["price"].to_numpy(dtype=float)
    y_pred = df["pred_model"].to_numpy(dtype=float)
    m = _compute_metrics(y_true, y_pred)
    resid = y_pred - y_true

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, resid, alpha=0.3)

    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Actual Sale Price (USD)")
    ax.set_ylabel("Residual (Predicted − Actual, USD)")
    ax.xaxis.set_major_formatter(_currency_formatter())
    ax.yaxis.set_major_formatter(_currency_formatter())

    lim = float(np.percentile(np.abs(resid), 99)) * 1.1
    ax.set_ylim(-lim, lim)

    title = (
        f"Residual vs Actual Price — Test Set  |  "
        f"MAE=${m['MAE']:,.0f}, RMSE={m['RMSE']:,.0f}, R²={m['R2']:.3f}"
    )
    if run_id:
        title += f"  (run {run_id})"
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plots] Saved {out_path}")


def plot_calibration(df: pd.DataFrame, out_path: Path, nbins: int = 20, run_id: str | None = None):
    """
    Simple reliability diagram: bin by predicted price, compare mean actual vs mean predicted.

    ECE_abs: average |mean_true - mean_pred| in USD (weighted by bin counts).
    ECE_rel: ECE_abs / mean(y_true) as a percentage.
    """
    y_true = df["price"].to_numpy(dtype=float)
    y_pred = df["pred_model"].to_numpy(dtype=float)

    # Equal-width bins over the predictions
    edges = np.linspace(y_pred.min(), y_pred.max(), nbins + 1)
    bin_idx = np.digitize(y_pred, edges) - 1

    bin_means_pred = []
    bin_means_true = []
    bin_counts = []

    for b in range(nbins):
        mask = bin_idx == b
        if not np.any(mask):
            continue
        bin_means_pred.append(float(np.mean(y_pred[mask])))
        bin_means_true.append(float(np.mean(y_true[mask])))
        bin_counts.append(int(np.sum(mask)))

    bin_means_pred = np.asarray(bin_means_pred)
    bin_means_true = np.asarray(bin_means_true)
    bin_counts = np.asarray(bin_counts, dtype=float)

    if bin_counts.sum() == 0:
        raise ValueError("No non-empty bins for calibration; check predictions.")

    diff = np.abs(bin_means_true - bin_means_pred)
    ece_abs = float(np.sum(diff * bin_counts) / bin_counts.sum())
    ece_rel = float(100.0 * ece_abs / np.mean(y_true))

    lo = float(min(bin_means_pred.min(), bin_means_true.min()))
    hi = float(max(bin_means_pred.max(), bin_means_true.max()))
    lo = max(0.0, lo)
    hi = hi * 1.05

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(bin_means_pred, bin_means_true)
    ax.plot([lo, hi], [lo, hi])

    ax.set_xlabel("Mean Predicted Price per Bin (USD)")
    ax.set_ylabel("Mean Actual Price per Bin (USD)")
    ax.xaxis.set_major_formatter(_currency_formatter())
    ax.yaxis.set_major_formatter(_currency_formatter())

    title = (
        f"Calibration — {nbins} Bins  |  "
        f"ECE_abs=${ece_abs:,.0f}, ECE_rel={ece_rel:.2f}%"
    )
    if run_id:
        title += f"  (run {run_id})"
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plots] Saved {out_path}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate standard plots for the CS542 Real Estate project."
    )
    parser.add_argument(
        "--cfg", default="configs/config.yaml", help="Path to master config YAML."
    )
    parser.add_argument(
        "--preds",
        default=None,
        help="Optional override path to test_preds.csv (defaults to <out_dir>/preds/test_preds.csv).",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Optional override for output figures directory (defaults to <out_dir>/figures).",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("scatter", help="Predicted vs actual scatter plot.")
    sub.add_parser("resid", help="Residual vs actual price plot.")

    calib_p = sub.add_parser("calib", help="Calibration / reliability diagram.")
    calib_p.add_argument("--bins", type=int, default=20, help="Number of bins (default: 20).")

    all_p = sub.add_parser("all", help="Generate all standard figures.")
    all_p.add_argument("--bins", type=int, default=20, help="Number of bins for calibration.")

    args = parser.parse_args(argv)

    cfg = _load_cfg(args.cfg)
    preds_path, fig_dir, run_id = _paths_from_cfg(cfg, args.preds, args.outdir)
    df = _load_preds(preds_path)

    if args.cmd == "scatter":
        plot_pred_vs_actual(df, fig_dir / "pred_vs_actual_test.png", run_id)
    elif args.cmd == "resid":
        plot_resid_vs_price(df, fig_dir / "resid_vs_price_test.png", run_id)
    elif args.cmd == "calib":
        plot_calibration(df, fig_dir / "calibration_test.png", nbins=args.bins, run_id=run_id)
    elif args.cmd == "all":
        plot_pred_vs_actual(df, fig_dir / "pred_vs_actual_test.png", run_id)
        plot_resid_vs_price(df, fig_dir / "resid_vs_price_test.png", run_id)
        plot_calibration(df, fig_dir / "calibration_test.png", nbins=args.bins, run_id=run_id)
    else:
        parser.error(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
