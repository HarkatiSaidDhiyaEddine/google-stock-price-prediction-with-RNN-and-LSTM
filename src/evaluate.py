"""Metrics and the naive persistence baseline.

Every number this project reports is paired with the baseline. A stock price is
extremely autocorrelated, so "tomorrow's price equals today's price" is a strong
predictor; a model that cannot beat it has not learned anything useful, and saying
otherwise would be misleading.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import pandas as pd

BASELINE_LABEL = "naive_baseline"


def format_metric_value(value: float | None) -> str:
    """Render a metric for the console, showing an undefined value as ``n/a``.

    Directional accuracy is undefined when a forecast never moves (see
    :func:`direction_accuracy`); printing ``nan`` or a misleading ``0.0000`` would
    invite the reader to treat a non-comparison as a comparison.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{float(value):.4f}"


class Metrics(NamedTuple):
    """Accuracy of a set of one-step-ahead forecasts."""

    rmse: float
    mae: float
    mape: float
    r2: float
    direction_accuracy: float
    sample_count: int

    def as_row(self) -> dict[str, float | int]:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "mape": self.mape,
            "r2": self.r2,
            "direction_accuracy": self.direction_accuracy,
            "samples": self.sample_count,
        }


def _validate_alignment(actual: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    actual_values = np.asarray(actual, dtype=float).ravel()
    predicted_values = np.asarray(predicted, dtype=float).ravel()

    if actual_values.shape != predicted_values.shape:
        raise ValueError(
            f"actual and predicted must have the same shape, got "
            f"{actual_values.shape} and {predicted_values.shape}"
        )
    if actual_values.size == 0:
        raise ValueError("cannot score an empty forecast")

    return actual_values, predicted_values


def direction_accuracy(
    actual: np.ndarray, predicted: np.ndarray, reference: np.ndarray
) -> float:
    """Share of days where the model got the *sign* of the move right.

    ``reference[t]`` is the last price known before forecast day ``t``, so both the
    actual move and the predicted move are measured from the same starting point.
    For predicting a price level this metric is the honest one: tracking the curve
    is easy, being consistently right about direction is not.

    Returns ``nan`` when the forecast never departs from the reference. Naive
    persistence is exactly that case, and its score is undefined rather than zero.
    """
    actual_values, predicted_values = _validate_alignment(actual, predicted)
    reference_values = np.asarray(reference, dtype=float).ravel()

    if reference_values.shape != actual_values.shape:
        raise ValueError(
            f"reference must have the same shape as actual, got "
            f"{reference_values.shape} and {actual_values.shape}"
        )

    actual_moves = np.sign(actual_values - reference_values)
    predicted_moves = np.sign(predicted_values - reference_values)

    if not np.any(predicted_moves):
        return float("nan")

    return float(np.mean(actual_moves == predicted_moves))


def up_day_share(actual: np.ndarray, reference: np.ndarray) -> float:
    """Share of days whose price ended above the last price known beforehand.

    This is the number :func:`direction_accuracy` has to be read against, and leaving it
    out is the easiest way to overstate a model. A forecast that always answers "up"
    scores *exactly* this share, so on a period where most days rose, a directional
    accuracy close to the up-day share is the base rate being collected for free rather
    than skill. On this project's test period the share is well above one half, which is
    precisely why the accuracy is printed beside it.
    """
    actual_values = np.asarray(actual, dtype=float).ravel()
    reference_values = np.asarray(reference, dtype=float).ravel()

    if actual_values.shape != reference_values.shape:
        raise ValueError(
            f"actual and reference must have the same shape, got "
            f"{actual_values.shape} and {reference_values.shape}"
        )
    if actual_values.size == 0:
        raise ValueError("cannot measure the up-day share of an empty period")

    return float(np.mean(actual_values > reference_values))


def regression_metrics(
    actual: np.ndarray, predicted: np.ndarray, reference: np.ndarray | None = None
) -> Metrics:
    """Compute RMSE, MAE, MAPE, R² and directional accuracy.

    ``reference`` supplies the previous price for directional accuracy. When it is
    omitted, direction is measured from the previous *actual* value.
    """
    actual_values, predicted_values = _validate_alignment(actual, predicted)
    residual = actual_values - predicted_values

    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))

    if np.any(actual_values == 0):
        raise ValueError("MAPE is undefined when an actual price is zero")
    mape = float(np.mean(np.abs(residual / actual_values)) * 100.0)

    total_variance = float(np.sum((actual_values - actual_values.mean()) ** 2))
    r2 = float(1.0 - np.sum(residual**2) / total_variance) if total_variance > 0 else float("nan")

    if reference is None:
        reference_values = np.concatenate([[actual_values[0]], actual_values[:-1]])
    else:
        reference_values = np.asarray(reference, dtype=float).ravel()

    return Metrics(
        rmse=rmse,
        mae=mae,
        mape=mape,
        r2=r2,
        direction_accuracy=direction_accuracy(actual_values, predicted_values, reference_values),
        sample_count=int(actual_values.size),
    )


def naive_persistence(previous_prices: np.ndarray) -> np.ndarray:
    """The baseline: predict that tomorrow's price equals today's.

    ``previous_prices[t]`` must be the last price observed before forecast day ``t``,
    which is the same alignment the model's inputs use. The returned array is
    therefore directly comparable with the model's predictions.
    """
    values = np.asarray(previous_prices, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("cannot build a baseline from an empty array")
    return values.copy()


def compare_to_baseline(
    actual: np.ndarray, predicted: np.ndarray, baseline: np.ndarray, reference: np.ndarray | None = None
) -> pd.DataFrame:
    """Metrics for the model and the baseline side by side, with the difference.

    A positive ``delta`` means the baseline is better (lower RMSE/MAE/MAPE), so the
    sign convention is stated explicitly rather than left to the reader.
    """
    model_metrics = regression_metrics(actual, predicted, reference)
    baseline_metrics = regression_metrics(actual, baseline, reference)

    rows = []
    for metric_name in ("rmse", "mae", "mape", "r2", "direction_accuracy"):
        model_value = getattr(model_metrics, metric_name)
        baseline_value = getattr(baseline_metrics, metric_name)
        rows.append(
            {
                "metric": metric_name,
                "model": model_value,
                BASELINE_LABEL: baseline_value,
                "delta": model_value - baseline_value,
            }
        )

    table = pd.DataFrame(rows)
    table.attrs["samples"] = model_metrics.sample_count
    table.attrs["beats_baseline_rmse"] = bool(model_metrics.rmse < baseline_metrics.rmse)
    # A baseline that never moves makes no directional call, so there is nothing for
    # the model to beat. Marked as undefined rather than False: claiming a directional
    # win over a forecast that makes no directional claim would be a fabrication.
    if math.isnan(baseline_metrics.direction_accuracy):
        table.attrs["beats_baseline_direction"] = None
    else:
        table.attrs["beats_baseline_direction"] = bool(
            model_metrics.direction_accuracy > baseline_metrics.direction_accuracy
        )

    # The base rate a permanent "up" forecast would score, carried on the table so the
    # rendered output can put it beside the model's directional accuracy. Without it the
    # accuracy reads as skill when it may be the up-day share of this particular period.
    table.attrs["up_day_share"] = (
        up_day_share(actual, reference) if reference is not None else None
    )
    return table


def format_table(table: pd.DataFrame, title: str) -> str:
    """Render a comparison table as plain text for the console."""
    samples = table.attrs.get("samples", 0)
    beats_rmse = table.attrs.get("beats_baseline_rmse", False)
    beats_direction = table.attrs.get("beats_baseline_direction")

    lines = [title, f"  forecasts scored: {samples}", ""]
    header = f"  {'metric':<20} {'model':>14} {BASELINE_LABEL:>14} {'delta':>14}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for _, row in table.iterrows():
        lines.append(
            f"  {row['metric']:<20} {format_metric_value(row['model']):>14} "
            f"{format_metric_value(row[BASELINE_LABEL]):>14} "
            f"{format_metric_value(row['delta']):>14}"
        )

    model_direction = float(table.loc[table["metric"] == "direction_accuracy", "model"].iloc[0])

    lines.append("")
    verdict = "BEATS" if beats_rmse else "does NOT beat"
    lines.append(f"  RMSE        : model {verdict} the naive baseline")

    if beats_direction is None:
        lines.append(
            f"  direction   : the naive baseline makes no directional call (it predicts"
        )
        lines.append(
            f"                exactly the previous price), so there is nothing to compare"
        )
        lines.append(
            f"                against; the model scored {format_metric_value(model_direction)}"
        )
    else:
        direction_verdict = "beats" if beats_direction else "does not beat"
        lines.append(f"  direction   : model {direction_verdict} the naive baseline")

    up_day = table.attrs.get("up_day_share")
    if up_day is not None:
        lines.append(
            f"  base rate   : {up_day:.4f} of these days closed above the previous close,"
        )
        lines.append(
            "                so a forecast that always answers 'up' scores exactly that."
        )
        lines.append(
            "                Directional accuracy is only skill to the extent it exceeds it."
        )
    return "\n".join(lines)
