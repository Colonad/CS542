# Phase 6–8 Summary — Run `2025-11-30_203109`

## Problem

Supervised regression task: predict residential **sale price** (`price`, USD) for U.S. properties, using structural features (bed, bath, house size, lot size) and location (city, state, ZIP), plus time-derived and neighborhood aggregate features.

## Data window & split protocol

- **Source CSV:** `data/raw/usa_real_estate.csv`
- **Observed sale dates:** 1901-01-01 → 2026-04-08
- **Temporal split (by `sold_date.year`):** train ≤ 2021, val = 2022, test ≥ 2023
- **Rows:** train = 50000, val = 460233, test = 499

## Models & metrics (baseline vs best)

| Model | Split | MAE | RMSE | R² |
|-------|-------|-----|------|----|
| ZIP×year median baseline | val | 203927.337 | 449075.788 | 0.369 |
| ZIP×year median baseline | test | 145896.556 | 279283.507 | 0.177 |
| Best model: **xgb** | val | 171879.692 | 363835.009 | 0.586 |
| Best model: **xgb** | test | 115261.168 | 184728.342 | 0.640 |

## Key insights

- The best model family on validation R² is **xgb**, with test MAE ≈ 115261.168 vs baseline MAE 145896.556.
- Tree-based and boosted models handle the mix of numeric and categorical features well once leak-safe neighborhood statistics are included.
- The log-transform of `price` combined with Duan smearing stabilizes training and yields more robust errors in high-price regions.

## Next steps

- Add cross-validation or rolling time-window validation to stress-test temporal stability.
- Run targeted ablations (e.g., remove neighbor features, remove log-transform) to quantify their impact.
- Investigate top error slices by state/ZIP (see `slices.json`) and decide whether to add region-specific features.
- Prepare final slides/report with figures: predicted vs actual, residuals vs price, and feature importance.
