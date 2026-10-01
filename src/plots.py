"""All figures, written to ``figures/``.

The matplotlib backend is forced to ``Agg`` so the pipeline runs headless from a
plain shell as well as from a notebook.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PRICE_LABEL = "Price (USD)"
DATE_LABEL = "Date"

MOVING_AVERAGE_WINDOWS: tuple[int, ...] = (100, 233)
FIGURE_DPI = 120


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)


def plot_price_history(dates: pd.Series, prices: np.ndarray, path: Path, column: str) -> None:
    """The whole price history, so the scale of the task is visible."""
    figure, axes = plt.subplots(figsize=(12, 5))
    axes.plot(dates, prices, linewidth=1.2, color="#1f77b4")
    axes.set_xlabel(DATE_LABEL)
    axes.set_ylabel(PRICE_LABEL)
    axes.set_title(f"GOOGL {column} price, full history")
    axes.grid(alpha=0.3)
    _save(figure, path)


def plot_price_with_moving_averages(
    dates: pd.Series, prices: np.ndarray, path: Path, column: str,
    windows: tuple[int, ...] = MOVING_AVERAGE_WINDOWS,
) -> None:
    """Price with its long moving averages.

    The notebook computed these indicators and then never used them; they are worth
    plotting because they show the trend the LSTM is trying to follow.
    """
    price_series = pd.Series(np.asarray(prices, dtype=float), index=pd.Index(dates, name=DATE_LABEL))
    figure, axes = plt.subplots(figsize=(12, 5))
    axes.plot(price_series.index, price_series.to_numpy(), linewidth=1.2, label=column, color="#1f77b4")

    colours = ["#ff7f0e", "#2ca02c", "#d62728"]
    for offset, window in enumerate(windows):
        moving_average = price_series.rolling(window).mean()
        axes.plot(
            price_series.index, moving_average.to_numpy(), linewidth=1.2,
            label=f"{window}-day MA", color=colours[offset % len(colours)],
        )

    axes.set_xlabel(DATE_LABEL)
    axes.set_ylabel(PRICE_LABEL)
    axes.set_title(f"GOOGL {column} with moving averages")
    axes.legend()
    axes.grid(alpha=0.3)
    _save(figure, path)


def plot_volume_history(
    dates: pd.Series, volumes: np.ndarray, path: Path, title: str
) -> None:
    """Traded volume over the whole history, in millions of shares.

    Volume is the one column of ``GOOGL.csv`` that never reaches the model; it is
    plotted purely to show how much the activity level changes across the period.
    """
    figure, axes = plt.subplots(figsize=(12, 4))
    axes.plot(dates, np.asarray(volumes, dtype=float) / 1e6, linewidth=1.0, color="#ff7f0e")
    axes.set_xlabel(DATE_LABEL)
    axes.set_ylabel("Volume (millions of shares)")
    axes.set_title(title)
    axes.grid(alpha=0.3)
    _save(figure, path)


def plot_return_distribution(returns: np.ndarray, path: Path, title: str) -> None:
    """Distribution of daily log returns, in percent.

    Returns are what the second prediction target models. Seeing their spread is the
    quickest way to understand why predicting a price level is a very different
    (and much easier) task than predicting the next move.
    """
    values = np.asarray(returns, dtype=float).ravel() * 100.0

    figure, axes = plt.subplots(figsize=(10, 5))
    axes.hist(values, bins=60, color="#2ca02c", edgecolor="white")
    axes.axvline(0.0, color="black", linewidth=1.0, linestyle=":")
    axes.axvline(
        float(values.mean()), color="#d62728", linewidth=1.5, linestyle="--",
        label=f"mean {values.mean():+.3f}%",
    )
    axes.set_xlabel("Daily log return (%)")
    axes.set_ylabel("Trading days")
    axes.set_title(title)
    axes.legend()
    axes.grid(alpha=0.3)
    _save(figure, path)


def plot_prediction_vs_actual(
    dates: pd.Series, actual: np.ndarray, predicted: np.ndarray, path: Path, title: str
) -> None:
    """Forecast against reality over the test period, on the real date axis."""
    figure, axes = plt.subplots(figsize=(12, 6))
    axes.plot(dates, actual, linewidth=1.5, label="actual", color="#1f77b4")
    axes.plot(dates, predicted, linewidth=1.5, label="predicted", color="#d62728")
    axes.set_xlabel(DATE_LABEL)
    axes.set_ylabel(PRICE_LABEL)
    axes.set_title(title)
    axes.legend()
    axes.grid(alpha=0.3)
    _save(figure, path)


def plot_residuals(
    dates: pd.Series, actual: np.ndarray, predicted: np.ndarray, path: Path, title: str
) -> None:
    """Prediction error over time, plus its distribution."""
    actual_values = np.asarray(actual, dtype=float).ravel()
    predicted_values = np.asarray(predicted, dtype=float).ravel()
    residual = actual_values - predicted_values

    figure, (time_axes, hist_axes) = plt.subplots(1, 2, figsize=(14, 5))
    time_axes.plot(dates, residual, linewidth=1.2, color="#9467bd")
    time_axes.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    time_axes.set_xlabel(DATE_LABEL)
    time_axes.set_ylabel("actual - predicted (USD)")
    time_axes.set_title("Prediction error over time")
    time_axes.grid(alpha=0.3)

    hist_axes.hist(residual, bins=40, color="#9467bd", edgecolor="white")
    hist_axes.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    hist_axes.set_xlabel("actual - predicted (USD)")
    hist_axes.set_ylabel("days")
    hist_axes.set_title(
        f"Error distribution (mean {residual.mean():+.2f}, std {residual.std():.2f})"
    )
    hist_axes.grid(alpha=0.3)

    figure.suptitle(title)
    _save(figure, path)


def plot_training_history(history, path: Path, title: str) -> None:
    """Training and validation loss per epoch."""
    train_loss = history.history.get("loss", [])
    validation_loss = history.history.get("val_loss", [])
    epochs = range(1, len(train_loss) + 1)

    figure, axes = plt.subplots(figsize=(10, 5))
    axes.plot(epochs, train_loss, linewidth=1.5, label="train loss")
    if validation_loss:
        axes.plot(epochs, validation_loss, linewidth=1.5, label="validation loss")
    axes.set_yscale("log")
    axes.set_xlabel("Epoch")
    axes.set_ylabel("MSE (log scale)")
    axes.set_title(title)
    axes.legend()
    axes.grid(alpha=0.3)
    _save(figure, path)
