# src/realestate/evaluate.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd
import yaml





__all__ = ["error_slices", "run"]

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


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
# Core slicing
# --------------------------------------------------------------------------- #
def error_slices(
    preds_csv: str | Path,
    *,
    group_cols: Iterable[str] = ("state", "zip_code"),
    actual_col: str = "price",
    pred_col: str = "pred_model",
    topk: int = 20,
) -> Dict[str, Dict[str, float]]:
    """
    Compute mean absolute error by each group column.

    Parameters
    ----------
    preds_csv : str | Path
        Path to predictions CSV (must contain `actual_col` and `pred_col`).
    group_cols : Iterable[str]
        Columns to group by. Columns that are missing in the CSV are skipped.
    actual_col : str
        Name of the actual/ground-truth price column (default 'price').
    pred_col : str
        Name of the model prediction column (default 'pred_model').
    topk : int
        Number of top worst groups (by MAE) to return.

    Returns
    -------
    dict
        { group_col: { group_value: mae, ... }, ... } with at most `topk` entries per group_col.
    """
    preds_csv = Path(preds_csv)
    if not preds_csv.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {preds_csv.resolve()}")

    df = pd.read_csv(preds_csv, low_memory=False)

    if actual_col not in df.columns or pred_col not in df.columns:
        raise KeyError(
            f"Missing required columns '{actual_col}' and/or '{pred_col}' in {preds_csv}"
        )

    # Absolute error
    y = df[actual_col].to_numpy(dtype=float)
    yhat = df[pred_col].to_numpy(dtype=float)
    abs_err = np.abs(y - yhat)
    df["_abs_err_"] = abs_err

    results: Dict[str, Dict[str, float]] = {}
    for g in group_cols:
        if g not in df.columns:
            # skip missing groups silently
            continue
        # MAE per group (drop NaN groups)
        grp = (
            df.dropna(subset=[g])
            .groupby(g)["_abs_err_"]
            .mean()
            .sort_values(ascending=False)
            .head(int(topk))
        )
        results[g] = {str(k): float(v) for k, v in grp.items()}

    # Housekeeping
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
        import matplotlib.pyplot as plt  # local import to keep dependency light
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

    # steps: read preds, compute slices, save slices, maybe 2 plots
    want_pred_vs_actual = bool(eval_cfg.get("plots", {}).get("pred_vs_actual", False))
    want_resid = bool(eval_cfg.get("plots", {}).get("residuals_vs_price", False))
    total_steps = 3 + (1 if want_pred_vs_actual else 0) + (1 if want_resid else 0)
    p = Prog(enabled=True, total=total_steps, desc="Evaluation")

    # Compute slices
    p.step("Reading predictions")
    slices = error_slices(
        preds_csv,
        group_cols=group_cols,
        actual_col=_get(cfg, ("target", "name"), "price"),
        pred_col="pred_model",
        topk=topk,
    )
    p.step("Computed error slices")

    # Save JSON
    (metrics_dir / "slices.json").write_text(json.dumps(slices, indent=2))
    p.step("Saved slices.json")

    # Optional quick plots
    if want_pred_vs_actual or want_resid:
        df = pd.read_csv(preds_csv, low_memory=False)
        _maybe_make_plots(
            df,
            figures_dir=figures_dir,
            actual_col=_get(cfg, ("target", "name"), "price"),
            pred_col="pred_model",
            cfg_eval=eval_cfg,
        )
        p.step("Saved plots")

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
