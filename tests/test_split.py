# tests/test_split.py
from __future__ import annotations

import pandas as pd
import pytest

from src.realestate.split import temporal_split


def test_temporal_split_basic_contract():
    df = pd.DataFrame(
        {
            "sold_date": pd.to_datetime(
                ["2022-12-31", "2023-01-01", "2023-06-15", "2022-01-05"]
            ),
            "price": [1, 2, 3, 4],
        }
    )
    train, test = temporal_split(df, "2023-01-01")

    # all train < cutoff; all test >= cutoff
    assert (train["sold_date"] < pd.Timestamp("2023-01-01")).all()
    assert (test["sold_date"] >= pd.Timestamp("2023-01-01")).all()

    # disjoint and complete coverage
    assert set(train.index).isdisjoint(test.index)
    assert len(train) + len(test) == len(df)

    # sorted output (default)
    assert train["sold_date"].is_monotonic_increasing
    assert test["sold_date"].is_monotonic_increasing


def test_temporal_split_parses_non_datetime_column():
    df = pd.DataFrame(
        {
            # intentionally strings
            "sold_date": ["2022-12-31", "2023-01-01"],
            "price": [1, 2],
        }
    )
    train, test = temporal_split(df, "2023-01-01")
    assert len(train) == 1 and len(test) == 1
    assert train.iloc[0]["sold_date"] == pd.Timestamp("2022-12-31")
    assert test.iloc[0]["sold_date"] == pd.Timestamp("2023-01-01")


def test_temporal_split_raises_on_missing_date_col():
    df = pd.DataFrame({"price": [1, 2]})
    with pytest.raises(KeyError):
        _ = temporal_split(df, "2023-01-01", date_col="sold_date")


def test_temporal_split_raises_on_unparseable_dates():
    df = pd.DataFrame({"sold_date": ["not-a-date", "also-bad"], "price": [1, 2]})
    with pytest.raises(ValueError):
        _ = temporal_split(df, "2023-01-01")


def test_temporal_split_raises_on_nat_dates():
    df = pd.DataFrame(
        {"sold_date": pd.to_datetime(["2022-12-31", None]), "price": [1, 2]}
    )
    with pytest.raises(ValueError):
        _ = temporal_split(df, "2023-01-01")


def test_temporal_split_empty_partition_error():
    # all dates are before cutoff -> test would be empty
    df_all_train = pd.DataFrame(
        {"sold_date": pd.to_datetime(["2020-01-01", "2020-02-01"]), "price": [1, 2]}
    )
    with pytest.raises(ValueError):
        _ = temporal_split(df_all_train, "2025-01-01")

    # all dates are after cutoff -> train would be empty
    df_all_test = pd.DataFrame(
        {"sold_date": pd.to_datetime(["2025-01-01", "2025-02-01"]), "price": [1, 2]}
    )
    with pytest.raises(ValueError):
        _ = temporal_split(df_all_test, "2025-01-01")


def test_temporal_split_custom_date_col_and_unsorted_output():
    df = pd.DataFrame(
        {
            "closing_date": pd.to_datetime(["2022-01-03", "2023-08-09", "2022-11-11"]),
            "row_id": [10, 11, 12],
        }
    )
    # Keep original order to ensure we can detect non-sorted behavior
    train, test = temporal_split(df, "2023-01-01", date_col="closing_date", sort_output=False)

    # contract still holds
    assert (train["closing_date"] < pd.Timestamp("2023-01-01")).all()
    assert (test["closing_date"] >= pd.Timestamp("2023-01-01")).all()

    # not necessarily sorted when sort_output=False (just check indices preserved subset-wise)
    assert list(train.index) == [0, 2]  # original order kept for those rows
    assert list(test.index) == [1]
