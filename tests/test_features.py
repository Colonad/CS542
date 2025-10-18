# tests/test_features.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.realestate.features import (
    basic_time_feats,
    years_since_prev,
    apply_engineered,
)


def test_basic_time_feats_columns_and_types():
    df = pd.DataFrame(
        {
            "sold_date": pd.to_datetime(["2023-02-01", "2024-03-15"]),
            "prev_sold_date": pd.to_datetime(["2020-02-01", None]),
            "price": [250_000, 310_000],
        }
    )
    out = basic_time_feats(df)

    # columns present
    assert {"year", "month", "years_since_prev_sale"}.issubset(out.columns)

    # types: year/month numeric (nullable int ok), years_since_prev_sale float
    assert pd.api.types.is_integer_dtype(out["year"])
    assert pd.api.types.is_integer_dtype(out["month"])
    assert pd.api.types.is_float_dtype(out["years_since_prev_sale"])

    # sanity on values
    assert out.loc[0, "year"] == 2023
    assert out.loc[0, "month"] == 2
    # first row has both dates -> finite non-negative years
    assert np.isfinite(out.loc[0, "years_since_prev_sale"])
    assert out.loc[0, "years_since_prev_sale"] >= 0
    # second row prev_sold_date is NaT -> NaN
    assert np.isnan(out.loc[1, "years_since_prev_sale"])


def test_basic_time_feats_requires_datetime():
    df = pd.DataFrame(
        {
            "sold_date": ["2023-02-01", "2024-03-15"],  # strings on purpose
            "price": [1, 2],
        }
    )
    with pytest.raises(TypeError):
        _ = basic_time_feats(df)


def test_years_since_prev_missing_column_creates_nan():
    df = pd.DataFrame(
        {"sold_date": pd.to_datetime(["2022-01-01", "2022-06-01"]), "price": [1, 2]}
    )
    out = years_since_prev(df)
    assert "years_since_prev_sale" in out.columns
    assert out["years_since_prev_sale"].isna().all()


def test_apply_engineered_rowwise_expressions():
    df = pd.DataFrame(
        {
            "bed": [3, 4],
            "bath": [2, 0],  # zero will be clipped to 1 in expr
            "acre_lot": [0.25, 0.5],
            "house_size": [1500, 2000],
            "sold_date": pd.to_datetime(["2023-01-01", "2023-02-01"]),
        }
    )

    cfg = {
        "features": {
            "engineered": [
                {"name": "bed_per_bath", "expr": "bed / np.clip(bath, 1, None)"},
                {"name": "lot_per_size", "expr": "acre_lot / np.clip(house_size, 1, None)"},
                {"name": "log_house_size", "expr": "np.log1p(house_size)"},
            ]
        }
    }

    out = apply_engineered(df, cfg)

    for col in ["bed_per_bath", "lot_per_size", "log_house_size"]:
        assert col in out.columns
        assert len(out[col]) == len(df)
        assert pd.api.types.is_float_dtype(out[col]) or pd.api.types.is_numeric_dtype(out[col])

    # specific checks
    # row 0: bed_per_bath = 3 / 2 = 1.5
    assert np.isclose(out.loc[0, "bed_per_bath"], 1.5, atol=1e-9)
    # row 1: bath clipped to 1 => 4 / 1 = 4.0
    assert np.isclose(out.loc[1, "bed_per_bath"], 4.0, atol=1e-9)
    # log_house_size monotone with house_size
    assert out["log_house_size"].iloc[1] > out["log_house_size"].iloc[0]


def test_apply_engineered_missing_column_raises():
    df = pd.DataFrame({"sold_date": pd.to_datetime(["2023-01-01"])})
    cfg = {"features": {"engineered": [{"name": "oops", "expr": "does_not_exist + 1"}]}}
    with pytest.raises(NameError):
        _ = apply_engineered(df, cfg)
