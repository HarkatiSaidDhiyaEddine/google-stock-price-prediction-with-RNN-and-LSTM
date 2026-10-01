"""Tests for the source-data loader.

``data/GOOGL.csv`` is the single source of truth, and every temporal guarantee in
this project rests on the loader sorting it once and rejecting anything malformed.
These tests pin that contract: chronological order, no duplicates, positive prices,
and a separate path for the columns the model never sees.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_loader import (
    DataValidationError,
    PriceSeries,
    load_auxiliary_frame,
    load_prices,
)

EXPECTED_ROW_COUNT = 4431
EXPECTED_FIRST_DATE = pd.Timestamp("2004-08-19")
EXPECTED_LAST_DATE = pd.Timestamp("2022-03-24")


@pytest.fixture(scope="module")
def prices() -> PriceSeries:
    return load_prices(config.DATA_PATH, config.TARGET_COLUMN)


def test_source_data_is_chronological(prices):
    assert prices.size == EXPECTED_ROW_COUNT
    assert prices.dates.is_monotonic_increasing
    assert (prices.values > 0).all()
    assert not prices.dates.duplicated().any()

    assert prices.dates.iloc[0] == EXPECTED_FIRST_DATE
    assert prices.dates.iloc[-1] == EXPECTED_LAST_DATE
    assert prices.column == config.TARGET_COLUMN


def test_loader_sorts_an_out_of_order_file(tmp_path):
    """The loader must impose the order, not trust the file to arrive sorted."""
    csv_path = tmp_path / "reversed.csv"
    csv_path.write_text(
        "Date,Close\n2020-01-03,3.0\n2020-01-01,1.0\n2020-01-02,2.0\n",
        encoding="utf-8",
    )

    series = load_prices(csv_path, "Close")

    assert series.dates.is_monotonic_increasing
    np.testing.assert_allclose(series.values, [1.0, 2.0, 3.0])


def test_loader_rejects_missing_column(tmp_path):
    csv_path = tmp_path / "broken.csv"
    csv_path.write_text("Date,Nope\n2020-01-01,1.0\n", encoding="utf-8")

    with pytest.raises(DataValidationError):
        load_prices(csv_path, "Close")


def test_loader_rejects_duplicate_dates(tmp_path):
    csv_path = tmp_path / "dupes.csv"
    csv_path.write_text(
        "Date,Close\n2020-01-01,1.0\n2020-01-01,2.0\n", encoding="utf-8"
    )

    with pytest.raises(DataValidationError):
        load_prices(csv_path, "Close")


def test_loader_rejects_a_non_positive_price(tmp_path):
    csv_path = tmp_path / "negative.csv"
    csv_path.write_text("Date,Close\n2020-01-01,1.0\n2020-01-02,-2.0\n", encoding="utf-8")

    with pytest.raises(DataValidationError):
        load_prices(csv_path, "Close")


def test_loader_rejects_a_null_price(tmp_path):
    csv_path = tmp_path / "null.csv"
    csv_path.write_text("Date,Close\n2020-01-01,1.0\n2020-01-02,\n", encoding="utf-8")

    with pytest.raises(DataValidationError):
        load_prices(csv_path, "Close")


def test_loader_rejects_an_unparseable_date(tmp_path):
    csv_path = tmp_path / "baddate.csv"
    csv_path.write_text("Date,Close\nnot-a-date,1.0\n", encoding="utf-8")

    with pytest.raises(DataValidationError):
        load_prices(csv_path, "Close")


def test_loader_reports_a_missing_file():
    with pytest.raises(DataValidationError):
        load_prices(config.DATA_DIR / "definitely_absent.csv", config.TARGET_COLUMN)


def test_auxiliary_frame_alignment_matches_the_price_series(prices):
    """Volume is plotted against the same date axis as the prices, so it must line up."""
    volume_frame = load_auxiliary_frame(config.DATA_PATH, [config.VOLUME_COLUMN])

    assert list(volume_frame.columns) == [config.DATE_COLUMN, config.VOLUME_COLUMN]
    assert len(volume_frame) == prices.size
    assert volume_frame[config.DATE_COLUMN].equals(prices.dates)


def test_auxiliary_frame_rejects_missing_and_null_columns(tmp_path):
    missing_path = tmp_path / "missing.csv"
    missing_path.write_text("Date,Close\n2020-01-01,1.0\n", encoding="utf-8")
    with pytest.raises(DataValidationError):
        load_auxiliary_frame(missing_path, ["Volume"])

    null_path = tmp_path / "nullcol.csv"
    null_path.write_text("Date,Volume\n2020-01-01,1.0\n2020-01-02,\n", encoding="utf-8")
    with pytest.raises(DataValidationError):
        load_auxiliary_frame(null_path, ["Volume"])


def test_auxiliary_loader_does_not_apply_the_positive_price_rule(tmp_path):
    """A zero could be meaningful in a non-price column; volume is not a price.

    The point is that ``load_auxiliary_frame`` never borrows the price-specific
    ``> 0`` rule, so a legitimately zero-valued auxiliary column is not a false alarm.
    """
    csv_path = tmp_path / "zero.csv"
    csv_path.write_text("Date,Volume\n2020-01-01,0\n2020-01-02,5\n", encoding="utf-8")

    volume_frame = load_auxiliary_frame(csv_path, ["Volume"])

    assert volume_frame["Volume"].tolist() == [0, 5]
