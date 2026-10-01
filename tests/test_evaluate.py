"""Tests for the metrics and the naive persistence baseline.

The baseline is the only reason the metrics in this project mean anything: every
number the model reports is only interesting relative to "tomorrow equals today".
These tests pin the arithmetic, the alignment, the treatment of undefined scores
and the sign convention of the comparison table.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluate import (
    BASELINE_LABEL,
    compare_to_baseline,
    direction_accuracy,
    format_metric_value,
    format_table,
    naive_persistence,
    regression_metrics,
    up_day_share,
)

ACTUAL = np.array([100.0, 110.0, 105.0, 120.0, 90.0])
PREVIOUS = np.array([95.0, 100.0, 110.0, 105.0, 120.0])


def test_perfect_forecast_scores_perfectly():
    metrics = regression_metrics(ACTUAL, ACTUAL, PREVIOUS)

    assert metrics.rmse == pytest.approx(0.0)
    assert metrics.mae == pytest.approx(0.0)
    assert metrics.mape == pytest.approx(0.0)
    assert metrics.r2 == pytest.approx(1.0)
    assert metrics.direction_accuracy == pytest.approx(1.0)
    assert metrics.sample_count == ACTUAL.size


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        regression_metrics(ACTUAL, ACTUAL[:-1])


def test_empty_forecast_is_rejected():
    with pytest.raises(ValueError):
        regression_metrics(np.array([]), np.array([]))


def test_mape_is_undefined_for_a_zero_price():
    actual = np.array([0.0, 10.0])
    predicted = np.array([1.0, 10.0])

    with pytest.raises(ValueError):
        regression_metrics(actual, predicted)


def test_naive_persistence_copies_the_previous_price():
    baseline = naive_persistence(PREVIOUS)

    np.testing.assert_allclose(baseline, PREVIOUS)
    # A copy, so a caller cannot mutate the caller's array through the baseline.
    assert baseline is not PREVIOUS


def test_direction_accuracy_counts_the_sign_of_the_move():
    # Measured from PREVIOUS these actual values move up, down, up, down, up.
    actual_prices = np.array([101.0, 99.0, 111.0, 104.0, 121.0])
    # A forecast one cent above the previous price calls every day up regardless.
    always_up = PREVIOUS + 0.01

    accuracy = direction_accuracy(actual_prices, always_up, PREVIOUS)

    # Three of the five days genuinely rose, so a permanent "up" call is right three times.
    assert accuracy == pytest.approx(3 / 5)


def test_direction_accuracy_is_undefined_when_the_forecast_never_moves():
    """Naive persistence makes no directional claim, so it has no direction score.

    Reporting this as 0.0 would hand the model a directional win it did not earn.
    """
    accuracy = direction_accuracy(ACTUAL, PREVIOUS, PREVIOUS)

    assert np.isnan(accuracy)


def test_format_metric_value_renders_undefined_values():
    assert format_metric_value(float("nan")) == "n/a"
    assert format_metric_value(None) == "n/a"
    assert format_metric_value(1.23456) == "1.2346"


def test_compare_to_baseline_reports_delta_and_verdicts():
    # The model is exactly the baseline: every delta must be zero, and it must be
    # reported as not beating the baseline on RMSE.
    table = compare_to_baseline(ACTUAL, PREVIOUS, PREVIOUS, PREVIOUS)

    rmse_row = table[table["metric"] == "rmse"].iloc[0]
    assert rmse_row["model"] == pytest.approx(rmse_row[BASELINE_LABEL])
    assert rmse_row["delta"] == pytest.approx(0.0)

    assert table.attrs["samples"] == ACTUAL.size
    assert table.attrs["beats_baseline_rmse"] is False
    # Undefined, not False: the baseline never moves, so it makes no directional call.
    assert table.attrs["beats_baseline_direction"] is None

    assert set(table["metric"]) == {"rmse", "mae", "mape", "r2", "direction_accuracy"}


def test_compare_to_baseline_flags_a_winning_model():
    # The model reproduces the actual series; the baseline only copies yesterday.
    table = compare_to_baseline(ACTUAL, ACTUAL, PREVIOUS, PREVIOUS)

    assert table.attrs["beats_baseline_rmse"] is True
    # The baseline still makes no directional call, so that comparison stays undefined.
    assert table.attrs["beats_baseline_direction"] is None

    rmse_row = table[table["metric"] == "rmse"].iloc[0]
    # A negative delta means the model's error is smaller, so the sign convention is
    # explicit rather than left to the reader.
    assert rmse_row["delta"] < 0.0


def test_up_day_share_counts_days_above_the_previous_close():
    # Measured from PREVIOUS these actual values move +5, +10, -5, +15, -30: three rise.
    share = up_day_share(ACTUAL, PREVIOUS)

    assert share == pytest.approx(3 / 5)


def test_up_day_share_equals_what_an_always_up_forecast_scores():
    """The definition of the base rate: a permanent "up" call scores exactly this."""
    always_up = PREVIOUS + 0.01

    accuracy_of_a_permanent_up_call = direction_accuracy(ACTUAL, always_up, PREVIOUS)

    assert accuracy_of_a_permanent_up_call == pytest.approx(up_day_share(ACTUAL, PREVIOUS))


def test_up_day_share_rejects_mismatched_lengths_and_empty_input():
    with pytest.raises(ValueError):
        up_day_share(ACTUAL, PREVIOUS[:-1])

    with pytest.raises(ValueError):
        up_day_share(np.array([]), np.array([]))


def test_compare_to_baseline_records_the_up_day_share():
    table = compare_to_baseline(ACTUAL, ACTUAL, PREVIOUS, PREVIOUS)

    assert table.attrs["up_day_share"] == pytest.approx(3 / 5)


def test_compare_to_baseline_leaves_the_base_rate_unset_without_a_reference():
    table = compare_to_baseline(ACTUAL, ACTUAL, PREVIOUS)

    assert table.attrs["up_day_share"] is None


def test_format_table_prints_the_always_up_base_rate():
    """Directional accuracy must never be printed without the number it has to beat."""
    table = compare_to_baseline(ACTUAL, ACTUAL, PREVIOUS, PREVIOUS)
    rendered = format_table(table, "model vs naive persistence:")

    assert "base rate" in rendered
    assert "0.6000" in rendered
    assert "always answers 'up'" in rendered


def test_format_table_prints_both_columns_and_explains_undefined_direction():
    table = compare_to_baseline(ACTUAL, ACTUAL, PREVIOUS, PREVIOUS)
    rendered = format_table(table, "model vs naive persistence:")

    assert "model vs naive persistence:" in rendered
    assert BASELINE_LABEL in rendered
    assert "rmse" in rendered
    assert "direction_accuracy" in rendered
    assert "no directional call" in rendered
    assert "n/a" in rendered

    # The undefined baseline score must show as `n/a` in the direction row itself,
    # not as a bare 0.0000 that would read like a real score of zero.
    direction_line = next(
        line for line in rendered.splitlines() if line.strip().startswith("direction_accuracy")
    )
    _, model_column, baseline_column = direction_line.split()[:3]
    assert model_column != "n/a"
    assert baseline_column == "n/a"
