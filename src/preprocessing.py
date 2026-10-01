"""Chronological splitting, scaling and windowing.

Every function here assumes the input is already sorted oldest-first (guaranteed
by :mod:`src.data_loader`). Nothing in this module is allowed to shuffle: the whole
point of the remake is that a window is 60 *consecutive* trading days.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

MINMAX_KIND = "minmax"
STANDARD_KIND = "standard"


class PreprocessingError(ValueError):
    """Raised when the inputs violate the pipeline's temporal assumptions."""


@dataclass(frozen=True)
class ChronologicalSplit:
    """A time-ordered split. ``test_frame`` starts exactly where ``train_frame`` ends."""

    train_frame: pd.DataFrame
    test_frame: pd.DataFrame
    split_index: int

    @property
    def train_size(self) -> int:
        return len(self.train_frame)

    @property
    def test_size(self) -> int:
        return len(self.test_frame)


def chronological_split(frame: pd.DataFrame, test_fraction: float) -> ChronologicalSplit:
    """Split ``frame`` by position, never by shuffling.

    The test set is the most recent ``test_fraction`` of the timeline, which is the
    only split that reflects how the model would actually be used.
    """
    if frame.empty:
        raise PreprocessingError("cannot split an empty frame")
    if not 0.0 < test_fraction < 1.0:
        raise PreprocessingError(f"test_fraction must be in (0, 1), got {test_fraction}")

    # ceil, so the split matches sklearn's train_test_split sizes (3,987 / 444)
    test_size = int(np.ceil(len(frame) * test_fraction))
    split_index = len(frame) - test_size

    if split_index <= 0 or test_size <= 0:
        raise PreprocessingError(
            f"test_fraction={test_fraction} leaves {split_index} train and "
            f"{test_size} test rows for {len(frame)} observations"
        )

    train_frame = frame.iloc[:split_index].reset_index(drop=True)
    test_frame = frame.iloc[split_index:].reset_index(drop=True)
    return ChronologicalSplit(train_frame, test_frame, split_index)


def fit_scaler(train_values: np.ndarray, kind: str = MINMAX_KIND):
    """Fit a scaler on the training values only.

    Fitting on train alone is the methodologically correct choice: it never lets the
    scaler observe a future price. The caller is responsible for reporting how far
    test values fall outside the range the scaler was fitted on.
    """
    column = np.asarray(train_values, dtype=float).reshape(-1, 1)
    if column.size == 0:
        raise PreprocessingError("cannot fit a scaler on an empty array")

    if kind == MINMAX_KIND:
        scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    elif kind == STANDARD_KIND:
        scaler = StandardScaler()
    else:
        raise PreprocessingError(f"unknown scaler kind: {kind!r}")

    scaler.fit(column)
    return scaler


def scale(scaler, values: np.ndarray) -> np.ndarray:
    """Apply a fitted scaler to a 1-D array and return a 1-D array."""
    column = np.asarray(values, dtype=float).reshape(-1, 1)
    return scaler.transform(column).ravel()


def inverse_scale(scaler, values: np.ndarray) -> np.ndarray:
    """Undo :func:`scale` for a 1-D array."""
    column = np.asarray(values, dtype=float).reshape(-1, 1)
    return scaler.inverse_transform(column).ravel()


def fraction_outside_training_range(scaler, values: np.ndarray) -> float:
    """Share of ``values`` outside the range a MinMax scaler was fitted on.

    A non-zero result is expected and meaningful under a chronological split: the
    test period can trade above or below anything seen during training. Reporting it
    is the honest alternative to silently clipping.
    """
    scaled = scale(scaler, values)
    return float(np.mean((scaled < 0.0) | (scaled > 1.0)))


def range_overshoot(scaler, values: np.ndarray) -> tuple[float, float]:
    """How *far* outside a MinMax scaler's fitted range ``values`` reach.

    Returns ``(mean_overshoot, max_overshoot)``, both in units of the fitted range:
    a value that maps to 1.39 overshoots by 0.39, and anything inside ``[0, 1]``
    overshoots by zero. Only meaningful for a bounded scaler -- a ``StandardScaler``
    has no range to leave -- so callers check the scaler kind first.

    The companion to :func:`fraction_outside_training_range`: that function says *how
    many* observations need extrapolation; this one says *how far*, and the two disagree
    in exactly the cases that matter. Four outlier days at 1.39 is a very different
    situation from 87% of the test period at 1.99, and a count alone cannot tell them
    apart. Like its companion, this raises on an empty array -- ``values`` must be the
    same non-empty slice the scaler is being asked about.
    """
    scaled = np.asarray(scale(scaler, values), dtype=float).ravel()
    overshoot = np.maximum(np.maximum(-scaled, scaled - 1.0), 0.0)
    return float(overshoot.mean()), float(overshoot.max())


def make_training_windows(series: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Turn a 1-D series into ``(samples, window, 1)`` inputs and 1-D targets.

    ``inputs[i]`` is ``series[i : i + window]`` and ``targets[i]`` is the very next
    value, ``series[i + window]``. Every window is consecutive by construction, and
    the sample count is derived from the data rather than hard-coded.
    """
    values = np.asarray(series, dtype=float).ravel()
    if window <= 0:
        raise PreprocessingError(f"window must be positive, got {window}")
    if len(values) <= window:
        raise PreprocessingError(
            f"need more than {window} observations to build windows, got {len(values)}"
        )

    sample_count = len(values) - window
    inputs = np.stack([values[i : i + window] for i in range(sample_count)])
    targets = values[window:]
    return inputs[..., np.newaxis], targets


def build_test_windows(context: np.ndarray, window: int, count: int) -> np.ndarray:
    """Build one window per forecast day from a single continuous history.

    ``context`` must be ``window + count`` long and may span the train/test boundary.
    The result has exactly ``count`` samples, so ``predictions[i]`` corresponds to
    forecast day ``i`` -- the alignment the original notebook lost when it compared
    shuffled rows positionally.
    """
    values = np.asarray(context, dtype=float).ravel()
    if len(values) != window + count:
        raise PreprocessingError(
            f"context must be window + count = {window + count} long, got {len(values)}"
        )

    inputs = np.stack([values[i : i + window] for i in range(count)])
    return inputs[..., np.newaxis]


def to_log_returns(prices: np.ndarray) -> np.ndarray:
    """``returns[i] = log(prices[i + 1] / prices[i])``.

    The result is one shorter than ``prices``: element ``i`` is the move *into* day
    ``i + 1``.
    """
    values = np.asarray(prices, dtype=float).ravel()
    if len(values) < 2:
        raise PreprocessingError("need at least two prices to compute returns")
    if np.any(values <= 0):
        raise PreprocessingError("prices must be positive to take log returns")
    return np.diff(np.log(values))


def to_price_changes(prices: np.ndarray) -> np.ndarray:
    """``changes[i] = prices[i + 1] - prices[i]``.

    Like :func:`to_log_returns`, element ``i`` is the move *into* day ``i + 1`` and the
    result is one shorter than ``prices``.

    Predicting the change instead of the level keeps the target roughly stationary
    around zero rather than trending upward, so a scaler fitted on the training changes
    stays broadly valid across the test period. It is *not* perfectly valid: the test
    period contains a handful of outlier days -- 2021-02-03, 2021-10-27, 2022-02-02 and
    2022-03-09 all rose by more than the largest daily rise in the whole training
    period -- and those few values do fall outside the fitted range. That is measured
    and reported (see :func:`fraction_outside_training_range` and
    :func:`range_overshoot`), not assumed away: the claim here is that the overflow is
    small and confined to outliers, unlike the level target where most of the test
    period lies outside the training range.
    """
    values = np.asarray(prices, dtype=float).ravel()
    if len(values) < 2:
        raise PreprocessingError("need at least two prices to compute changes")
    return np.diff(values)


def apply_price_changes(previous_prices: np.ndarray, changes: np.ndarray) -> np.ndarray:
    """``price[t] = previous_prices[t] + predicted_change[t]``."""
    previous = np.asarray(previous_prices, dtype=float).ravel()
    change_values = np.asarray(changes, dtype=float).ravel()
    if previous.shape != change_values.shape:
        raise PreprocessingError(
            f"shape mismatch: previous_prices {previous.shape} vs changes {change_values.shape}"
        )
    return previous + change_values


def reconstruct_prices(previous_prices: np.ndarray, log_returns: np.ndarray) -> np.ndarray:
    """``price[t] = previous_prices[t] * exp(predicted_log_return[t])``.

    ``previous_prices[t]`` is the last price known before forecast day ``t``.
    """
    previous = np.asarray(previous_prices, dtype=float).ravel()
    returns = np.asarray(log_returns, dtype=float).ravel()
    if previous.shape != returns.shape:
        raise PreprocessingError(
            f"shape mismatch: previous_prices {previous.shape} vs returns {returns.shape}"
        )
    return previous * np.exp(returns)
