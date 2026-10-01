"""Tests for the chronological pipeline.

Two tests carry the weight here.

:func:`test_window_last_value_correlates_with_target` asserts
``corr(window[-1], target) > 0.9`` -- exactly the property the original notebook
violated (it measured -0.0248), and the check that fails loudly the moment anybody
reintroduces a shuffle.

:func:`test_price_change_target_needs_far_less_extrapolation_than_the_price_level`
and :func:`test_price_change_overshoot_is_confined_to_outliers` pin *why* the
``price_change`` target is the one that works: a growth-trending price level forces a
train-fitted scaler to extrapolate for most of the test period, while day-to-day
differences leave it almost entirely inside the range it was fitted on. The second of
the pair exists because a fraction of affected days cannot distinguish "four outlier
days" from "the target stopped being stationary" -- only the magnitude can.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from src import config
from src.data_loader import load_prices
from src.preprocessing import (
    PreprocessingError,
    apply_price_changes,
    build_test_windows,
    chronological_split,
    fit_scaler,
    fraction_outside_training_range,
    inverse_scale,
    make_training_windows,
    range_overshoot,
    reconstruct_prices,
    scale,
    to_log_returns,
    to_price_changes,
)

MAX_TRADING_GAP_DAYS = 7  # generous: covers weekends and public holidays

# The extrapolation each target actually needs, pinned so the comparison is a
# regression test rather than a narrative. `price_change` is NOT extrapolation-free:
# four 2021-2022 outlier days escape the training range. It must, however, stay an
# order of magnitude closer to that range than `price`, which leaves it for most of
# the test set.
MAX_CHANGE_EXTRAPOLATION_DAYS = 0.02
MIN_LEVEL_EXTRAPOLATION_DAYS = 0.50
MIN_EXTRAPOLATION_RATIO = 10.0


@pytest.fixture(scope="module")
def prices():
    return load_prices(config.DATA_PATH, config.TARGET_COLUMN)


@pytest.fixture(scope="module")
def split(prices):
    return chronological_split(prices.frame, config.TEST_FRACTION)


def test_split_is_positional_and_contiguous(split, prices):
    # ceil matches sklearn's train_test_split: 4431 rows -> 3987 train / 444 test
    expected_test_size = int(np.ceil(prices.size * config.TEST_FRACTION))

    assert split.train_size + split.test_size == prices.size
    assert split.test_size == expected_test_size
    assert split.train_size == prices.size - expected_test_size
    assert (split.train_size, split.test_size) == (3987, 444)

    last_train_date = split.train_frame["Date"].iloc[-1]
    first_test_date = split.test_frame["Date"].iloc[0]
    assert last_train_date < first_test_date

    # The test frame begins exactly at the row after the train frame ends.
    assert prices.frame["Date"].iloc[split.split_index] == first_test_date
    assert prices.frame["Date"].iloc[split.split_index - 1] == last_train_date


def test_train_and_test_do_not_overlap(split):
    train_dates = set(split.train_frame["Date"])
    test_dates = set(split.test_frame["Date"])
    assert train_dates.isdisjoint(test_dates)


def test_window_count_matches_formula(prices):
    series = prices.values
    inputs, targets = make_training_windows(series, config.WINDOW_SIZE)
    expected_samples = len(series) - config.WINDOW_SIZE

    assert inputs.shape == (expected_samples, config.WINDOW_SIZE, 1)
    assert targets.shape == (expected_samples,)
    # No hard-coded 3927 anywhere: the sample count follows the data.
    assert inputs.shape[0] == expected_samples


def test_windows_use_consecutive_trading_days(prices):
    """A shuffled window would jump years between consecutive rows."""
    dates = prices.dates.to_numpy()
    series = prices.values
    inputs, targets = make_training_windows(series, config.WINDOW_SIZE)

    # Sample windows spread across the whole timeline, not just the first few.
    sample_indices = np.linspace(0, len(inputs) - 1, num=25, dtype=int)
    for index in sample_indices:
        window_dates = dates[index : index + config.WINDOW_SIZE]
        gaps = np.diff(window_dates).astype("timedelta64[D]").astype(int)
        assert gaps.max() <= MAX_TRADING_GAP_DAYS, (
            f"window {index} spans a {gaps.max()}-day jump, so its rows are not "
            f"consecutive trading days"
        )
        # The window's own values must match the corresponding source rows.
        np.testing.assert_allclose(inputs[index, :, 0], series[index : index + config.WINDOW_SIZE])
        assert targets[index] == series[index + config.WINDOW_SIZE]


def test_window_last_value_correlates_with_target(prices):
    """The regression test for the whole project.

    With consecutive days, the last price in a window is almost perfectly correlated
    with the next price. With shuffled rows the correlation collapses to ~0 and the
    model can only learn the mean -- the exact symptom of the original notebook.
    """
    series = prices.values
    inputs, targets = make_training_windows(series, config.WINDOW_SIZE)
    correlation = float(np.corrcoef(inputs[:, -1, 0], targets)[0, 1])

    assert correlation > 0.9, (
        f"corr(window[-1], target) = {correlation:+.4f}. A value near zero means the "
        f"windows are not in chronological order -- check for a shuffling split."
    )


def test_build_test_windows_aligns_each_prediction_with_its_day(prices):
    series = prices.values
    window = config.WINDOW_SIZE
    count = 30
    context = series[-count - window :]

    inputs = build_test_windows(context, window, count)

    assert inputs.shape == (count, window, 1)
    for day in range(count):
        np.testing.assert_allclose(inputs[day, :, 0], context[day : day + window])
        # The most recent observation in the window is the previous day's price,
        # so prediction[day] is a genuine one-step-ahead forecast.
        assert inputs[day, -1, 0] == context[day + window - 1]

    assert inputs[0, -1, 0] == series[-count - 1]


def test_build_test_windows_rejects_wrong_context_length():
    with pytest.raises(PreprocessingError):
        build_test_windows(np.arange(50, dtype=float), window=60, count=10)


def test_scaler_is_fitted_on_train_only_and_reports_extrapolation(prices, split):
    train_values = split.train_frame[config.TARGET_COLUMN].to_numpy(dtype=float)
    test_values = split.test_frame[config.TARGET_COLUMN].to_numpy(dtype=float)

    scaler = fit_scaler(train_values)

    assert scaler.data_min_[0] == pytest.approx(train_values.min())
    assert scaler.data_max_[0] == pytest.approx(train_values.max())
    # The scaler never saw a test price, so the training max would be wrong if it had.
    assert scaler.data_max_[0] != pytest.approx(prices.values.max())

    fraction = fraction_outside_training_range(scaler, test_values)
    assert 0.0 <= fraction <= 1.0

    scaled_train = scale(scaler, train_values)
    assert scaled_train.min() == pytest.approx(0.0)
    assert scaled_train.max() == pytest.approx(1.0)

    restored = inverse_scale(scaler, scaled_train)
    np.testing.assert_allclose(restored, train_values, rtol=1e-9)


def test_log_returns_round_trip(prices):
    values = prices.values
    returns = to_log_returns(values)

    assert returns.shape == (len(values) - 1,)

    # Reconstruct every price after the first from the previous price and the return.
    reconstructed = reconstruct_prices(values[:-1], returns)
    np.testing.assert_allclose(reconstructed, values[1:], rtol=1e-9)


def test_log_returns_reject_non_positive_prices():
    with pytest.raises(PreprocessingError):
        to_log_returns(np.array([1.0, 0.0, 2.0]))


def test_price_changes_round_trip(prices):
    """``P[t] + (P[t+1] - P[t])`` must reproduce the next price exactly."""
    values = prices.values
    changes = to_price_changes(values)

    assert changes.shape == (len(values) - 1,)

    reconstructed = apply_price_changes(values[:-1], changes)
    np.testing.assert_allclose(reconstructed, values[1:], rtol=1e-12)


def test_apply_price_changes_rejects_shape_mismatch():
    with pytest.raises(PreprocessingError):
        apply_price_changes(np.arange(5, dtype=float), np.arange(4, dtype=float))


def _test_period_changes(prices, split) -> tuple[np.ndarray, np.ndarray]:
    """The training and test slices of the price-change series, aligned to the split.

    ``changes[i]`` is the move into day ``i + 1``, so the training slice is one shorter
    than the training frame and the test slice starts at the last training row -- the
    same alignment ``run_price_change_target`` uses.
    """
    changes = to_price_changes(prices.values)
    return (
        changes[: split.split_index - 1],
        changes[split.split_index - 1 : split.split_index - 1 + split.test_size],
    )


def test_price_change_target_needs_far_less_extrapolation_than_the_price_level(prices, split):
    """Why the ``price_change`` target exists -- measured, not asserted.

    A price *level* trends upward across these eighteen years, so a scaler fitted on the
    training slice must encode most of the test period outside the range it ever saw
    (87.4% of days). Day-to-day changes oscillate around zero instead of drifting, so the
    same train-only scaler covers all but a handful of test days (0.9%).

    The difference target is *not* extrapolation-free, and claiming it was would be the
    same kind of unearned claim this project exists to avoid. Four specific 2021-2022
    days -- 2021-02-03, 2021-10-27, 2022-02-02 and 2022-03-09 -- rose by more than any
    single day in the 2004-2018 training window, so they do overshoot it. The bounds
    below are therefore pinned to the measured behaviour instead of to zero.
    """
    train_changes, test_changes = _test_period_changes(prices, split)
    train_prices = split.train_frame[config.TARGET_COLUMN].to_numpy(dtype=float)

    change_extrapolation = fraction_outside_training_range(
        fit_scaler(train_changes), test_changes
    )
    price_extrapolation = fraction_outside_training_range(
        fit_scaler(train_prices), prices.values[split.split_index:]
    )

    assert change_extrapolation <= MAX_CHANGE_EXTRAPOLATION_DAYS, (
        f"{change_extrapolation:.1%} of test-period price changes fall outside the range "
        f"seen during training; the difference target should be almost fully covered"
    )
    # The level target really does have to extrapolate, so the comparison is not vacuous.
    assert price_extrapolation >= MIN_LEVEL_EXTRAPOLATION_DAYS
    assert change_extrapolation * MIN_EXTRAPOLATION_RATIO <= price_extrapolation


def test_price_change_overshoot_is_confined_to_outliers(prices, split):
    """How *far* the change target overshoots -- the part a day count cannot express.

    Four days leave the fitted range, and each does so by a small fraction of that
    range. If the overshoot ever blew up, the difference target would be degrading back
    into the problem it was introduced to fix, and the fraction of affected days would
    not reveal it: a handful of days can leave the range by any margin at all.
    """
    train_changes, test_changes = _test_period_changes(prices, split)

    mean_overshoot, max_overshoot = range_overshoot(
        fit_scaler(train_changes), test_changes
    )

    assert 0.0 < max_overshoot < 0.5, (
        f"the largest test-period change overshoots the training range by "
        f"{max_overshoot:.3f} of that range"
    )
    # A mean near zero is the point: the overflow sits on a few outlier days rather than
    # being spread across the test period.
    assert mean_overshoot < 0.05


def test_range_overshoot_is_zero_inside_the_fitted_range():
    scaler = fit_scaler(np.array([0.0, 10.0, 20.0]))

    mean_overshoot, max_overshoot = range_overshoot(scaler, np.array([0.0, 5.0, 20.0]))

    assert mean_overshoot == pytest.approx(0.0)
    assert max_overshoot == pytest.approx(0.0)


def test_range_overshoot_measures_the_distance_beyond_the_range():
    """A value mapping to 1.5 overshoots by 0.5; one mapping to -0.25 overshoots by 0.25."""
    scaler = fit_scaler(np.array([0.0, 10.0]))

    mean_overshoot, max_overshoot = range_overshoot(scaler, np.array([15.0, -2.5]))

    assert max_overshoot == pytest.approx(0.5)
    assert mean_overshoot == pytest.approx((0.5 + 0.25) / 2)


def test_every_prediction_target_is_fully_configured():
    """Every configured target must be usable, or the pipeline fails mid-run.

    This checks the configuration side. The runner side lives in ``src.run_pipeline``,
    which cannot be imported here without dragging TensorFlow into a pure-NumPy test
    module, so that module asserts the matching invariant at import time instead.
    """
    assert set(config.PREDICTION_TARGETS) == {
        config.PRICE_TARGET,
        config.PRICE_CHANGE_TARGET,
        config.LOG_RETURN_TARGET,
    }
    assert set(config.PREDICTION_TARGETS) <= set(config.SCALER_KIND_BY_TARGET)

    # A plain price level must be MinMax-scaled; a return is stationary, so standard.
    assert config.SCALER_KIND_BY_TARGET[config.PRICE_TARGET] == "minmax"
    assert config.SCALER_KIND_BY_TARGET[config.PRICE_CHANGE_TARGET] == "minmax"
    assert config.SCALER_KIND_BY_TARGET[config.LOG_RETURN_TARGET] == "standard"


def test_model_paths_are_distinct_per_target():
    paths = {target: config.model_path_for(target) for target in config.PREDICTION_TARGETS}

    assert len(set(paths.values())) == len(paths)
    for target, path in paths.items():
        assert path.parent == config.MODEL_DIR
        assert target in path.name


def test_only_the_price_level_is_persistent():
    """The shuffle check applies to the price level and nowhere else.

    ``corr(window[-1], target)`` must be near +1 for a persistent target and is ~0 by
    construction for a stationary one, so the two must not be confused. Marking a
    difference target persistent would raise the anti-shuffle alarm on a healthy run;
    marking the price level stationary would silently stop checking for the original
    bug.
    """
    assert config.PERSISTENT_TARGETS <= set(config.PREDICTION_TARGETS)
    assert config.target_is_persistent(config.PRICE_TARGET)
    assert not config.target_is_persistent(config.PRICE_CHANGE_TARGET)
    assert not config.target_is_persistent(config.LOG_RETURN_TARGET)


def test_split_rejects_invalid_fraction(prices):
    for bad_fraction in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(PreprocessingError):
            chronological_split(prices.frame, bad_fraction)


def _called_function_names(module_path: Path) -> list[str]:
    """Every function name called in a module, read from the syntax tree.

    Parsing the AST instead of grepping the text means documentation such as
    "nothing in this module is allowed to shuffle" is not mistaken for a call to a
    shuffling routine.
    """
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    names: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            names.append(func.attr)
        elif isinstance(func, ast.Name):
            names.append(func.id)

    return names


def test_no_shuffling_calls_exist_in_the_source_tree():
    """Guard against anyone reintroducing the original bug."""
    forbidden_calls = {"train_test_split", "shuffle", "permutation", "sample"}
    offenders = []

    for module_path in sorted(Path(config.PROJECT_ROOT, "src").glob("*.py")):
        for name in _called_function_names(module_path):
            if name in forbidden_calls:
                offenders.append(f"{module_path.name}: {name}()")

    assert not offenders, f"shuffling construct(s) called in src/: {offenders}"


def test_no_shuffle_keyword_arguments_are_used():
    """``shuffle=True`` on a call is the same bug wearing a different hat."""
    offenders = []

    for module_path in sorted(Path(config.PROJECT_ROOT, "src").glob("*.py")):
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if keyword.arg == "shuffle":
                    offenders.append(
                        f"{module_path.name}: shuffle={ast.unparse(keyword.value)}"
                    )

    assert not offenders, f"shuffle keyword argument(s) found in src/: {offenders}"
