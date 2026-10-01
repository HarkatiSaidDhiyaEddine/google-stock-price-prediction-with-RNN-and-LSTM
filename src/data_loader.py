"""Load and validate the raw GOOGL price file.

The only source of truth is ``data/GOOGL.csv``. Everything downstream consumes the
``PriceSeries`` returned here, so the temporal ordering is established exactly once
and can be asserted in tests.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATE_COLUMN


class DataValidationError(ValueError):
    """Raised when the source CSV does not satisfy the pipeline's assumptions."""


@dataclass(frozen=True)
class PriceSeries:
    """A single price column with its dates, guaranteed to be in ascending order."""

    frame: pd.DataFrame
    column: str

    @property
    def dates(self) -> pd.Series:
        return self.frame[DATE_COLUMN]

    @property
    def values(self) -> np.ndarray:
        return self.frame[self.column].to_numpy(dtype=float)

    @property
    def size(self) -> int:
        return len(self.frame)

    def describe(self) -> str:
        first = self.dates.iloc[0].date()
        last = self.dates.iloc[-1].date()
        return (
            f"{self.column}: {self.size} rows, {first} -> {last}, "
            f"min {self.values.min():.2f}, max {self.values.max():.2f}"
        )


def _read_columns(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    """Read ``columns`` plus the date, sorted oldest-first, with no null dates.

    Shared by :func:`load_prices` and :func:`load_auxiliary_frame`. Every check that
    depends on a column's *meaning* -- prices must be positive, for instance -- is
    left to the caller; this only guarantees a parseable, unique, ascending date.
    """
    if not path.exists():
        raise DataValidationError(
            f"source data not found at {path}. The pipeline expects data/GOOGL.csv."
        )

    frame = pd.read_csv(path)
    missing = {DATE_COLUMN, *columns} - set(frame.columns)
    if missing:
        raise DataValidationError(
            f"{path.name} is missing required column(s): {sorted(missing)}. "
            f"Found: {list(frame.columns)}"
        )

    frame = frame[[DATE_COLUMN, *columns]].copy()
    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], errors="coerce")

    if frame[DATE_COLUMN].isna().any():
        raise DataValidationError(f"some dates in {path.name} could not be parsed")

    if frame[DATE_COLUMN].duplicated().any():
        duplicate_count = int(frame[DATE_COLUMN].duplicated().sum())
        raise DataValidationError(
            f"{path.name} contains {duplicate_count} duplicate date(s)"
        )

    frame = frame.sort_values(DATE_COLUMN).reset_index(drop=True)

    if not frame[DATE_COLUMN].is_monotonic_increasing:
        raise DataValidationError("dates are not strictly increasing after sorting")

    return frame


def load_prices(path: Path, column: str) -> PriceSeries:
    """Read ``path`` and return the ``column`` price series sorted oldest-first.

    Raises:
        DataValidationError: if the file is missing, the column is absent, the
            prices contain nulls or non-positive values, or the dates are not
            unique.
    """
    frame = _read_columns(path, [column])

    if frame[column].isna().any():
        null_count = int(frame[column].isna().sum())
        raise DataValidationError(f"{column}: {null_count} null price(s) in {path.name}")

    if (frame[column] <= 0).any():
        raise DataValidationError(f"{column}: non-positive price found in {path.name}")

    return PriceSeries(frame=frame, column=column)


def load_auxiliary_frame(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    """Read columns the model never sees, for the EDA figures only.

    Kept separate from :func:`load_prices` so the price-specific validation (strictly
    positive values) cannot be applied to a column like volume, and so nothing on the
    modelling path can depend on it.
    """
    frame = _read_columns(path, columns)

    null_columns = [name for name in columns if frame[name].isna().any()]
    if null_columns:
        raise DataValidationError(f"{path.name}: null value(s) in {null_columns}")

    return frame
