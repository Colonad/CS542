# tests/test_temporal_split.py

from pathlib import Path

import pandas as pd

from realestate import io as io_mod
from realestate.train_sweep import _load_yaml, temporal_three_way_split


def test_temporal_three_way_split_zero_leakage():
    """
    Definition of Done (Phase 9):

    - Temporal split must enforce:
        train <= train_max_year
        val   == val_year
        test  >= test_min_year
    - No raw row appears in more than one split (index-based disjointness).
    """
    cfg = _load_yaml("configs/config.yaml")
    data_csv = Path(cfg["paths"]["data_csv"])

    df = io_mod.load_data(data_csv, cfg=cfg)

    split_cfg = cfg.get("split", {})
    train_max = int(split_cfg.get("train_max_year", 2022))
    val_year = int(split_cfg.get("val_year", 2023))
    test_min = int(split_cfg.get("test_min_year", 2024))

    tr, va, te = temporal_three_way_split(
        df,
        date_col="sold_date",
        train_max_year=train_max,
        val_year=val_year,
        test_min_year=test_min,
    )

    # Sanity: splits should not be completely empty for this dataset
    assert not tr.empty, "Train split is empty; check temporal split config vs data."
    assert not va.empty, "Validation split is empty; check temporal split config vs data."
    assert not te.empty, "Test split is empty; check temporal split config vs data."

    # Year-based invariants
    tr_years = tr["sold_date"].dt.year
    va_years = va["sold_date"].dt.year
    te_years = te["sold_date"].dt.year

    assert (tr_years <= train_max).all()
    assert (va_years == val_year).all()
    assert (te_years >= test_min).all()

    # Zero leakage: no row index appears in more than one split
    idx_tr = set(tr.index)
    idx_va = set(va.index)
    idx_te = set(te.index)

    assert idx_tr.isdisjoint(idx_va)
    assert idx_tr.isdisjoint(idx_te)
    assert idx_va.isdisjoint(idx_te)
