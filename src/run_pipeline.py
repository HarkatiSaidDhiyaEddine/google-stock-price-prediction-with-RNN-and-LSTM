"""End-to-end pipeline: load, split, scale, window, train, evaluate, plot.

Run it from the project root with::

    python -m src.run_pipeline

Three prediction targets are trained and scored side by side. They differ only in
what the network is asked to output; the windowing, the architecture, the split and
the baseline are identical, so the comparison is fair.

``price``
    The original tutorial's task: predict tomorrow's closing price as a level. A
    MinMax scaler fitted on the training slice only is asked to encode test prices
    that may lie outside the range it ever saw, so this target is expected to lose
    resolution. It is kept because it is the task the original notebook set out to
    solve, and because measuring how badly it degrades is the point.

``price_change``
    Predict tomorrow's closing price *minus* today's, then rebuild the level with
    ``P[t+1] = P[t] + d``. Differences oscillate around zero instead of trending
    upward, so a scaler fitted on the training period keeps covering almost all of the
    test period. It is not a perfect fit: four outlier days in 2021-2022 rose by more
    than any day in training and do overshoot the fitted range. The pipeline prints
    both the share of days affected and how far they overshoot, because a fraction
    alone cannot describe the difference between this target and ``price``.

``log_return``
    Predict tomorrow's log return, then rebuild the price with
    ``P[t+1] = P[t] * exp(r)``. The most honest target: there is no price level to
    copy, so directional accuracy becomes a meaningful score.

Nothing here shuffles, and no number is reported without its baseline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config, plots
from src.data_loader import PriceSeries, load_auxiliary_frame, load_prices
from src.evaluate import (
    BASELINE_LABEL,
    Metrics,
    compare_to_baseline,
    format_metric_value,
    format_table,
    naive_persistence,
    regression_metrics,
    up_day_share,
)
from src.preprocessing import (
    MINMAX_KIND,
    ChronologicalSplit,
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
from src.train import seed_everything, train_model

DIVIDER = "=" * 78
FIRST_DAYS_SHOWN = 5
LAST_DAYS_SHOWN = 5


@dataclass(frozen=True)
class TargetRun:
    """Everything one prediction target produced, ready for reporting or plotting."""

    prediction_target: str
    scaler_kind: str
    train_samples: int
    validation_samples: int
    test_days: int
    epochs_run: int
    best_validation_loss: float
    window_correlation: float
    outside_training_range: float
    # How far outside the fitted range the test values reach, in units of that range.
    # The count above says how many days need extrapolation; these say how badly, and
    # the two disagree exactly where it matters -- see :func:`range_overshoot`.
    mean_range_overshoot: float
    max_range_overshoot: float
    predicted_std: float
    actual_std: float
    price_correlation: float
    # Spread of the network's raw output in the target's own units (dollars of level,
    # dollars of change, or log return). The price-level standard deviation above is
    # inherited from the reference prices for the two difference targets, so it cannot
    # tell "the model forecasts" apart from "the model copies yesterday"; this can.
    predicted_target_std: float
    actual_target_std: float
    # How much of the target's own spread the network reproduced, and whether that clears
    # MIN_TARGET_VARIANCE_RATIO. False means the network is emitting an effectively
    # constant value -- the MSE-optimal answer for a target it cannot predict, and the
    # thing that must be reported rather than dressed up.
    target_variance_ratio: float
    forecasts_target_variation: bool
    # Share of test days that closed above the previous close. A forecast that always
    # answers "up" scores exactly this, so directional accuracy must be read against it.
    up_day_share: float
    # Whether the previous value of this target predicts the next one, which decides how
    # corr(window[-1], target) should be read.
    is_persistent_target: bool
    dates: pd.Series
    predicted_prices: np.ndarray
    actual_prices: np.ndarray
    baseline_prices: np.ndarray
    metrics_table: pd.DataFrame
    model_metrics: Metrics
    history: object


def _report_run(run: TargetRun) -> None:
    """Print the per-target block: shapes, diagnostics, metrics, sample forecasts."""
    print(f"\n{DIVIDER}")
    print(f"prediction target: {run.prediction_target}  (scaler: {run.scaler_kind})")
    print(DIVIDER)
    print(f"  training samples      : {run.train_samples}")
    print(f"  validation samples    : {run.validation_samples}  (chronological tail of train)")
    print(f"  test days             : {run.test_days}")
    print(f"  epochs actually run   : {run.epochs_run}")
    print(f"  best validation loss  : {run.best_validation_loss:.6f}")
    print(f"  corr(window[-1], target) = {run.window_correlation:+.4f}")
    # One number, two opposite readings. For the price level it is the anti-shuffle
    # check and must be near +1; for a stationary target it is ~0 by construction and
    # says nothing about ordering. Spell out which applies, or the same value looks
    # like a problem in one block and a success in another.
    if run.is_persistent_target:
        print("    (price level: near +1 required. A value near zero is the shuffled-window)")
        print("    (signature -- the original notebook measured -0.0248 here.)")
    else:
        print("    (stationary target: ~0 is expected by construction and says nothing)")
        print("    (about window order, so this number is not the shuffle check here.)")
    if run.outside_training_range > 0.0:
        print(
            f"  test values outside training range: "
            f"{run.outside_training_range * 100:.1f}%  (expected under a chronological split)"
        )
        print(
            f"  how far outside that range: mean {run.mean_range_overshoot:.4f}, "
            f"max {run.max_range_overshoot:.4f}  (in units of the fitted range)"
        )
    print(
        f"  predicted std = ${run.predicted_std:.2f}   "
        f"actual std = ${run.actual_std:.2f}   "
        f"corr(predicted, actual) = {run.price_correlation:+.4f}"
    )
    # The price std above is inherited from the reference prices for the two
    # difference targets; this is the network's own output spread in the units it was
    # actually asked to predict, so it can tell a forecast apart from a copy.
    print(
        f"  model output std in target space = {run.predicted_target_std:.6g}   "
        f"actual target std = {run.actual_target_std:.6g}"
    )
    if run.forecasts_target_variation:
        print(
            f"  target-space variance ratio = {run.target_variance_ratio:.4f}  "
            f"(the network does vary its output with the input)"
        )
    else:
        print(
            f"  target-space variance ratio = {run.target_variance_ratio:.4f}  "
            f"(EFFECTIVELY CONSTANT -- see the diagnosis below)"
        )
    print(
        f"  directional accuracy = {format_metric_value(run.model_metrics.direction_accuracy)}   "
        f"up-day share = {run.up_day_share:.4f}   "
        f"(always-'up' would score exactly the up-day share)"
    )
    print()
    print(format_table(run.metrics_table, "model vs naive persistence:"))
    print()
    print(f"  first {FIRST_DAYS_SHOWN} test days (predicted / actual / baseline):")
    for date, predicted, actual, baseline in list(
        zip(run.dates, run.predicted_prices, run.actual_prices, run.baseline_prices)
    )[:FIRST_DAYS_SHOWN]:
        print(
            f"    {pd.Timestamp(date).date()}  predicted {predicted:9.2f}"
            f"  actual {actual:9.2f}  baseline {baseline:9.2f}"
        )
    print(f"  last {LAST_DAYS_SHOWN} test days:")
    for date, predicted, actual, baseline in list(
        zip(run.dates, run.predicted_prices, run.actual_prices, run.baseline_prices)
    )[-LAST_DAYS_SHOWN:]:
        print(
            f"    {pd.Timestamp(date).date()}  predicted {predicted:9.2f}"
            f"  actual {actual:9.2f}  baseline {baseline:9.2f}"
        )


def _finalise_run(
    prediction_target: str,
    scaler_kind: str,
    train_samples: int,
    validation_samples: int,
    history,
    window_correlation: float,
    outside_training_range: float,
    range_overshoot_values: tuple[float, float],
    predicted_targets: np.ndarray,
    actual_targets: np.ndarray,
    dates: pd.Series,
    predicted_prices: np.ndarray,
    actual_prices: np.ndarray,
    baseline_prices: np.ndarray,
    reference_prices: np.ndarray,
) -> TargetRun:
    """Assemble a :class:`TargetRun`, computing every derived statistic once."""
    metrics_table = compare_to_baseline(
        actual_prices, predicted_prices, baseline_prices, reference_prices
    )
    model_metrics = regression_metrics(actual_prices, predicted_prices, reference_prices)

    predicted_target_values = np.asarray(predicted_targets, dtype=float).ravel()
    actual_target_values = np.asarray(actual_targets, dtype=float).ravel()
    if predicted_target_values.shape != actual_target_values.shape:
        raise ValueError(
            f"predicted and actual targets must align, got "
            f"{predicted_target_values.shape} and {actual_target_values.shape}"
        )

    # How much of the target's own spread the network reproduced. This is the only
    # statistic here that separates "the model forecasts the next move" from "the model
    # emits the target's average and the price is rebuilt around it", because the
    # reconstructed price level inherits its variance from the reference prices.
    predicted_target_std = float(np.std(predicted_target_values))
    actual_target_std = float(np.std(actual_target_values))
    target_variance_ratio = (
        predicted_target_std / actual_target_std if actual_target_std > 0.0 else 0.0
    )

    run = TargetRun(
        prediction_target=prediction_target,
        scaler_kind=scaler_kind,
        train_samples=train_samples,
        validation_samples=validation_samples,
        test_days=int(len(actual_prices)),
        epochs_run=len(history.history["loss"]),
        best_validation_loss=float(min(history.history["val_loss"])),
        window_correlation=window_correlation,
        outside_training_range=outside_training_range,
        mean_range_overshoot=range_overshoot_values[0],
        max_range_overshoot=range_overshoot_values[1],
        predicted_std=float(np.std(predicted_prices)),
        actual_std=float(np.std(actual_prices)),
        price_correlation=float(np.corrcoef(predicted_prices, actual_prices)[0, 1]),
        predicted_target_std=predicted_target_std,
        actual_target_std=actual_target_std,
        target_variance_ratio=target_variance_ratio,
        forecasts_target_variation=target_variance_ratio >= MIN_TARGET_VARIANCE_RATIO,
        up_day_share=up_day_share(actual_prices, reference_prices),
        is_persistent_target=config.target_is_persistent(prediction_target),
        dates=dates,
        predicted_prices=predicted_prices,
        actual_prices=actual_prices,
        baseline_prices=baseline_prices,
        metrics_table=metrics_table,
        model_metrics=model_metrics,
        history=history,
    )
    return run


class PipelineError(RuntimeError):
    """Raised when the pipeline produces a result that cannot be trusted."""


CONSTANT_PREDICTION_THRESHOLD_USD = 50.0
# A model whose output spans less than this share of the target's own spread has not
# learned to forecast it. Used for the two difference targets, whose reconstructed
# price level keeps its variance even when the model emits a constant.
MIN_TARGET_VARIANCE_RATIO = 0.05

TRAINING_ARGUMENTS: dict[str, object] = {
    "validation_fraction": config.VALIDATION_FRACTION,
    "epochs": config.EPOCHS,
    "batch_size": config.BATCH_SIZE,
    "patience": config.PATIENCE,
    "cell_activation": config.CELL_ACTIVATION,
    "hidden_units": config.HIDDEN_UNITS,
    "dropout": config.DROPOUT_RATE,
    "window": config.WINDOW_SIZE,
}


def _outside_training_range(scaler, scaler_kind: str, values: np.ndarray) -> float:
    """Only a MinMax scaler has a bounded range worth checking against."""
    if scaler_kind != MINMAX_KIND:
        return 0.0
    return fraction_outside_training_range(scaler, values)


def _range_overshoot(
    scaler, scaler_kind: str, values: np.ndarray
) -> tuple[float, float]:
    """Only a MinMax scaler has a bounded range that can be overshot at all."""
    if scaler_kind != MINMAX_KIND:
        return 0.0, 0.0
    return range_overshoot(scaler, values)


def _full_series_window_correlation(series: np.ndarray) -> float:
    """``corr(window[-1], target)`` across the whole timeline.

    This is the number the original notebook measured at -0.0248. Anything close to
    zero means the windows are not in chronological order.
    """
    inputs, targets = make_training_windows(series, config.WINDOW_SIZE)
    return float(np.corrcoef(inputs[:, -1, 0], targets)[0, 1])


def run_price_target(prices: PriceSeries, split: ChronologicalSplit, verbose: int = 2) -> TargetRun:
    """Predict tomorrow's closing price. This is the original tutorial's task."""
    window = config.WINDOW_SIZE
    column = config.TARGET_COLUMN
    all_prices = prices.values
    split_index = split.split_index
    test_days = split.test_size

    train_prices = split.train_frame[column].to_numpy(dtype=float)
    test_prices = split.test_frame[column].to_numpy(dtype=float)

    scaler_kind = config.SCALER_KIND_BY_TARGET[config.PRICE_TARGET]
    scaler = fit_scaler(train_prices, scaler_kind)

    x_train, y_train = make_training_windows(scale(scaler, train_prices), window)

    model, history, (x_val, _) = train_model(
        inputs=x_train,
        targets=y_train,
        model_path=config.model_path_for(config.PRICE_TARGET),
        verbose=verbose,
        **TRAINING_ARGUMENTS,
    )

    # The test context starts `window` days before the split so that the first test
    # window is 60 real, consecutive trading days ending on the last training day.
    context = all_prices[split_index - window : split_index + test_days]
    scaled_context = scale(scaler, context)
    x_test = build_test_windows(scaled_context, window, test_days)
    predicted_prices = inverse_scale(scaler, model.predict(x_test, verbose=0).ravel())

    # The last price known before each test day: the naive forecast, and the
    # reference point for directional accuracy.
    reference_prices = all_prices[split_index - 1 : split_index + test_days - 1]

    return _finalise_run(
        prediction_target=config.PRICE_TARGET,
        scaler_kind=scaler_kind,
        train_samples=len(x_train),
        validation_samples=len(x_val),
        history=history,
        window_correlation=_full_series_window_correlation(all_prices),
        outside_training_range=_outside_training_range(scaler, scaler_kind, test_prices),
        range_overshoot_values=_range_overshoot(scaler, scaler_kind, test_prices),
        predicted_targets=predicted_prices,
        actual_targets=test_prices,
        dates=split.test_frame[config.DATE_COLUMN].reset_index(drop=True),
        predicted_prices=predicted_prices,
        actual_prices=test_prices,
        baseline_prices=naive_persistence(reference_prices),
        reference_prices=reference_prices,
    )


def run_price_change_target(prices: PriceSeries, split: ChronologicalSplit, verbose: int = 2) -> TargetRun:
    """Predict tomorrow's closing price as a *difference*, then rebuild the level.

    Same forecast as :func:`run_price_target`, different parameterisation. The
    network outputs ``P[t+1] - P[t]`` and the price is rebuilt as ``P[t] + d``.

    This is the fix for the one thing that genuinely degrades the level target: a
    price series trends upward over eighteen years, so a MinMax scaler fitted on the
    training period maps later prices outside ``[0, 1]`` and the model has to
    extrapolate. Day-to-day differences oscillate around zero rather than trending, so
    a scaler fitted on training differences covers the test period almost exactly: 0.9%
    of test days fall outside it, against 87% for the level target. Those 0.9% are
    genuine outliers -- four 2021-2022 up-days larger than any day in training -- and
    not a drift, which is the distinction a lone fraction cannot express, hence the
    count *and* the magnitude both being printed.
    """
    window = config.WINDOW_SIZE
    all_prices = prices.values
    split_index = split.split_index
    test_days = split.test_size

    # changes[i] is the move into day i + 1, so this array is one shorter than prices.
    all_changes = to_price_changes(all_prices)
    train_changes = all_changes[: split_index - 1]

    scaler_kind = config.SCALER_KIND_BY_TARGET[config.PRICE_CHANGE_TARGET]
    scaler = fit_scaler(train_changes, scaler_kind)

    x_train, y_train = make_training_windows(scale(scaler, train_changes), window)

    model, history, (x_val, _) = train_model(
        inputs=x_train,
        targets=y_train,
        model_path=config.model_path_for(config.PRICE_CHANGE_TARGET),
        verbose=verbose,
        **TRAINING_ARGUMENTS,
    )

    # Changes known at the start of the test period come from the training tail.
    context = all_changes[split_index - 1 - window :]
    scaled_context = scale(scaler, context)
    x_test = build_test_windows(scaled_context, window, test_days)
    predicted_changes = inverse_scale(scaler, model.predict(x_test, verbose=0).ravel())

    reference_prices = all_prices[split_index - 1 : split_index + test_days - 1]
    predicted_prices = apply_price_changes(reference_prices, predicted_changes)

    # Only the test-period changes are worth checking: the training ones defined the
    # scaler's range by construction.
    test_changes = all_changes[split_index - 1 : split_index - 1 + test_days]

    return _finalise_run(
        prediction_target=config.PRICE_CHANGE_TARGET,
        scaler_kind=scaler_kind,
        train_samples=len(x_train),
        validation_samples=len(x_val),
        history=history,
        window_correlation=_full_series_window_correlation(all_changes),
        outside_training_range=_outside_training_range(scaler, scaler_kind, test_changes),
        range_overshoot_values=_range_overshoot(scaler, scaler_kind, test_changes),
        predicted_targets=predicted_changes,
        actual_targets=test_changes,
        dates=split.test_frame[config.DATE_COLUMN].reset_index(drop=True),
        predicted_prices=predicted_prices,
        actual_prices=split.test_frame[config.TARGET_COLUMN].to_numpy(dtype=float),
        baseline_prices=naive_persistence(reference_prices),
        reference_prices=reference_prices,
    )


def run_log_return_target(prices: PriceSeries, split: ChronologicalSplit, verbose: int = 2) -> TargetRun:
    """Predict tomorrow's log return, then rebuild the price from it."""
    window = config.WINDOW_SIZE
    all_prices = prices.values
    split_index = split.split_index
    test_days = split.test_size

    # returns[i] is the move into day i + 1, so this array is one shorter than prices.
    all_returns = to_log_returns(all_prices)
    train_returns = all_returns[: split_index - 1]

    scaler_kind = config.SCALER_KIND_BY_TARGET[config.LOG_RETURN_TARGET]
    scaler = fit_scaler(train_returns, scaler_kind)

    x_train, y_train = make_training_windows(scale(scaler, train_returns), window)

    model, history, (x_val, _) = train_model(
        inputs=x_train,
        targets=y_train,
        model_path=config.model_path_for(config.LOG_RETURN_TARGET),
        verbose=verbose,
        **TRAINING_ARGUMENTS,
    )

    # Returns known at the start of the test period come from the training tail.
    context = all_returns[split_index - 1 - window :]
    scaled_context = scale(scaler, context)
    x_test = build_test_windows(scaled_context, window, test_days)
    predicted_returns = inverse_scale(scaler, model.predict(x_test, verbose=0).ravel())

    reference_prices = all_prices[split_index - 1 : split_index + test_days - 1]
    predicted_prices = reconstruct_prices(reference_prices, predicted_returns)
    test_returns = all_returns[split_index - 1 : split_index - 1 + test_days]

    return _finalise_run(
        prediction_target=config.LOG_RETURN_TARGET,
        scaler_kind=scaler_kind,
        train_samples=len(x_train),
        validation_samples=len(x_val),
        history=history,
        window_correlation=_full_series_window_correlation(all_returns),
        outside_training_range=_outside_training_range(scaler, scaler_kind, test_returns),
        range_overshoot_values=_range_overshoot(scaler, scaler_kind, test_returns),
        predicted_targets=predicted_returns,
        actual_targets=test_returns,
        dates=split.test_frame[config.DATE_COLUMN].reset_index(drop=True),
        predicted_prices=predicted_prices,
        actual_prices=split.test_frame[config.TARGET_COLUMN].to_numpy(dtype=float),
        baseline_prices=naive_persistence(reference_prices),
        reference_prices=reference_prices,
    )


TARGET_RUNNERS = {
    config.PRICE_TARGET: run_price_target,
    config.PRICE_CHANGE_TARGET: run_price_change_target,
    config.LOG_RETURN_TARGET: run_log_return_target,
}

# A target listed in config with no runner would only surface as a KeyError halfway
# through a training run, after the first model had already been fitted. Checked at
# import time so the gap is impossible to ship.
_MISSING_RUNNERS = sorted(set(config.PREDICTION_TARGETS) - set(TARGET_RUNNERS))
if _MISSING_RUNNERS:
    raise RuntimeError(
        f"config.PREDICTION_TARGETS lists {_MISSING_RUNNERS} with no runner in "
        f"TARGET_RUNNERS. Every configured target must be wired end to end."
    )


def _write_eda_figures(prices: PriceSeries) -> None:
    """Figures that describe the data, independent of any model.

    Four panels, all produced from ``data/GOOGL.csv`` alone: the price history, the
    price against its long moving averages, traded volume, and the distribution of
    daily log returns. Volume and returns are never fed to the network -- they are
    here because the second and third prediction targets are about *returns*, and the
    spread of that distribution is the clearest way to see why predicting a price
    level and predicting the next move are different problems.
    """
    plots.plot_price_history(
        prices.dates,
        prices.values,
        config.FIGURES_DIR / "price_history.png",
        prices.column,
    )
    plots.plot_price_with_moving_averages(
        prices.dates,
        prices.values,
        config.FIGURES_DIR / "price_moving_averages.png",
        prices.column,
    )

    volume_frame = load_auxiliary_frame(config.DATA_PATH, [config.VOLUME_COLUMN])
    plots.plot_volume_history(
        volume_frame[config.DATE_COLUMN],
        volume_frame[config.VOLUME_COLUMN].to_numpy(dtype=float),
        config.FIGURES_DIR / "volume_history.png",
        "GOOGL traded volume",
    )
    plots.plot_return_distribution(
        to_log_returns(prices.values),
        config.FIGURES_DIR / "return_distribution.png",
        "Distribution of daily GOOGL log returns",
    )


def _write_run_figures(run: TargetRun) -> None:
    """Forecast, error and learning-curve figures for one prediction target."""
    name = run.prediction_target
    plots.plot_prediction_vs_actual(
        run.dates,
        run.actual_prices,
        run.predicted_prices,
        config.FIGURES_DIR / f"prediction_vs_actual_{name}.png",
        f"One-step-ahead forecast, target = {name} (chronological split)",
    )
    plots.plot_residuals(
        run.dates,
        run.actual_prices,
        run.predicted_prices,
        config.FIGURES_DIR / f"residuals_{name}.png",
        f"Prediction error, target = {name}",
    )
    plots.plot_training_history(
        run.history,
        config.FIGURES_DIR / f"training_history_{name}.png",
        f"Training and validation loss, target = {name}",
    )


def _assert_price_predictions_vary(runs: list[TargetRun]) -> None:
    """Fail loudly if a target reproduced the original constant-prediction symptom.

    This is the regression test for the shuffled split. The original notebook emitted
    the same ~$595 forecast for every one of its 444 test days -- a predicted price
    standard deviation of $1.23 in sample and $5.81 across the whole test set, against
    an actual spread of $2,933. Any target whose predicted *price* level is that flat
    again has that bug back, and the run is aborted instead of reported.

    Only the price level is checked. The two difference targets rebuild their price as
    ``previous_close + change`` (or ``x exp(return)``), so their price spread is
    inherited from the reference prices and stays in the hundreds of dollars even when
    the network emits a constant -- checking it would prove nothing. Those are reported
    by :func:`_report_flat_targets` instead.
    """
    offenders = [
        run.prediction_target
        for run in runs
        if run.predicted_std < CONSTANT_PREDICTION_THRESHOLD_USD
    ]
    if offenders:
        raise PipelineError(
            f"the predicted price level is effectively constant for {offenders}: the "
            f"price standard deviation is below ${CONSTANT_PREDICTION_THRESHOLD_USD:.2f} "
            f"over the whole test set. That is the original notebook's symptom, and it "
            f"means the windows carry no information about the target. Check that the "
            f"split is chronological and that corr(window[-1], target) is high."
        )


def _flat_targets(runs: list[TargetRun]) -> list[TargetRun]:
    """Targets whose network output is effectively constant in the target's own units."""
    return [
        run
        for run in runs
        if run.actual_target_std > 0.0 and not run.forecasts_target_variation
    ]


def _report_flat_targets(runs: list[TargetRun]) -> None:
    """Report, without aborting, any target the network predicts as a constant.

    This is not a crash: for a stationary target a constant is the *correct* answer to
    the question an MSE loss asks. Daily returns are close to unpredictable, so the
    MSE-optimal forecast is the target's mean, and a network that outputs that mean has
    converged rather than failed. What must not happen is calling it a success. The
    reconstructed price curve still tracks the test period -- but only because it is
    anchored to the previous close, which is information the naive baseline had for
    free, so the tracking is not evidence that the model forecast anything.

    Aborting instead would be worse than reporting: the run would die after spending
    half an hour training, and a reader would learn nothing about *why* it collapsed.
    """
    flat = _flat_targets(runs)
    if not flat:
        print()
        print("  every target's own output varies: no network collapsed to a constant")
        return

    print(f"\n{DIVIDER}")
    print("diagnosis: network output effectively constant in target space")
    print(DIVIDER)
    for run in flat:
        print(
            f"  {run.prediction_target:<12} output std {run.predicted_target_std:.6g} vs "
            f"actual target std {run.actual_target_std:.6g}  "
            f"(ratio {run.target_variance_ratio:.4f}, below {MIN_TARGET_VARIANCE_RATIO:.0%})"
        )
    print()
    print("  What this means, and what it does not:")
    print("    - The network has not learned to forecast day-to-day variation in these")
    print("      targets. Predicting the training mean is the MSE-optimal answer when the")
    print("      target is close to unpredictable, which daily returns are, so this is")
    print("      convergence to the mean rather than a bug.")
    print("    - The price curve still tracks the test period because the level is")
    print("      rebuilt from the previous close. That tracking is inherited, not earned:")
    print("      the naive baseline is handed the same anchor for free, which is why the")
    print("      RMSE verdict is the one that counts.")
    print("    - It is NOT the shuffled-split symptom. That collapses the predicted price")
    print("      level itself, which _assert_price_predictions_vary already fails on.")


def _print_summary(runs: list[TargetRun]) -> None:
    """One table comparing all targets, plus the reference to the broken notebook."""
    print(f"\n{DIVIDER}")
    print("summary")
    print(DIVIDER)

    header = (
        f"  {'target':<12} {'pred std':>10} {'actual std':>11} {'corr':>8} "
        f"{'RMSE':>10} {'naive':>10} {'beats':>6} {'dir':>7} {'up days':>8} {'out var':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for run in runs:
        rmse_row = run.metrics_table[run.metrics_table["metric"] == "rmse"].iloc[0]
        direction_row = run.metrics_table[
            run.metrics_table["metric"] == "direction_accuracy"
        ].iloc[0]
        beats = "yes" if run.metrics_table.attrs["beats_baseline_rmse"] else "no"
        print(
            f"  {run.prediction_target:<12} {run.predicted_std:>10.2f} {run.actual_std:>11.2f} "
            f"{run.price_correlation:>8.4f} {rmse_row['model']:>10.2f} "
            f"{rmse_row[BASELINE_LABEL]:>10.2f} {beats:>6} "
            f"{format_metric_value(direction_row['model']):>7} "
            f"{run.up_day_share:>8.4f} {run.target_variance_ratio:>8.4f}"
        )

    print()
    print("  reference point -- the original notebook, with a shuffled split, produced")
    print("  predictions spanning $5.81 across the entire test set while the actual")
    print("  prices spanned $2933 (corr(window[-1], target) = -0.0248).")
    for run in runs:
        ratio = run.predicted_std / 5.81 if run.predicted_std else 0.0
        print(
            f"    {run.prediction_target:<12} predicted std ${run.predicted_std:.2f} "
            f"= {ratio:.0f}x the old $5.81 spread"
        )

    print()
    print("  Every metric above is paired with the naive persistence baseline.")
    print("  'beats'   RMSE only. A model that loses to persistence has not learned")
    print("            something the previous day's price did not already tell it.")
    print("  'dir'     share of test days whose move sign the model got right, and")
    print("  'up days' the share a permanent always-'up' forecast would score. A")
    print("            directional accuracy near the up-day share is the base rate of")
    print("            this test period, not evidence of skill.")
    print("  'out var' the network's own output spread as a fraction of the target's.")
    print("            Below 0.05 the network emitted an effectively constant value --")
    print("            the MSE-optimal forecast for a target it cannot predict -- and")
    print("            the reconstructed price curve tracks the test period only")
    print("            because it is anchored to the previous close.")


def main(verbose: int = 2) -> list[TargetRun]:
    """Run the whole pipeline and return one :class:`TargetRun` per prediction target."""
    config.ensure_directories()
    seed_everything(config.SEED)

    print(DIVIDER)
    print("GOOGL next-day price forecasting -- stacked LSTM")
    print(
        f"  seed={config.SEED}  window={config.WINDOW_SIZE}  epochs<={config.EPOCHS}  "
        f"batch={config.BATCH_SIZE}  activation={config.CELL_ACTIVATION}"
    )
    print(DIVIDER)

    prices = load_prices(config.DATA_PATH, config.TARGET_COLUMN)
    print(f"  data  : {prices.describe()}")

    split = chronological_split(prices.frame, config.TEST_FRACTION)
    train_dates = split.train_frame[config.DATE_COLUMN]
    test_dates = split.test_frame[config.DATE_COLUMN]
    print(
        f"  train : {train_dates.iloc[0].date()} -> {train_dates.iloc[-1].date()} "
        f"({split.train_size} rows)"
    )
    print(
        f"  test  : {test_dates.iloc[0].date()} -> {test_dates.iloc[-1].date()} "
        f"({split.test_size} rows)"
    )
    print("  split : chronological, no shuffling at any stage")

    _write_eda_figures(prices)

    runs: list[TargetRun] = []
    for prediction_target in config.PREDICTION_TARGETS:
        run = TARGET_RUNNERS[prediction_target](prices, split, verbose=verbose)
        _report_run(run)
        _write_run_figures(run)
        runs.append(run)

    _assert_price_predictions_vary(runs)
    _report_flat_targets(runs)
    _print_summary(runs)

    print(f"\n  figures   : {config.FIGURES_DIR}")
    print(f"  checkpoints: {config.MODEL_DIR}")
    return runs


if __name__ == "__main__":
    main()
