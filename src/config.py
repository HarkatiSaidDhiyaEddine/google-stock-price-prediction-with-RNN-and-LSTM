"""Single source of truth for paths, model hyper-parameters and seeds."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
MODEL_DIR: Path = PROJECT_ROOT / "models"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"

DATA_PATH: Path = DATA_DIR / "GOOGL.csv"

DATE_COLUMN: str = "Date"
TARGET_COLUMN: str = "Close"
# Read only for the EDA figures. It never reaches the model.
VOLUME_COLUMN: str = "Volume"

WINDOW_SIZE: int = 60
TEST_FRACTION: float = 0.10
VALIDATION_FRACTION: float = 0.10

BATCH_SIZE: int = 32
PATIENCE: int = 10

# `LSTM_EPOCHS` lets a quick verification run cap training without touching code.
EPOCHS: int = int(os.environ.get("LSTM_EPOCHS", "100"))

SEED: int = 42

# Keras LSTM cell-output activation. "tanh" is the Keras default; the original
# notebook overrode it with "relu", which is unbounded and non-standard inside a
# recurrence. Kept configurable so the two can be measured against each other.
CELL_ACTIVATION: str = "tanh"
HIDDEN_UNITS: tuple[int, ...] = (60, 60, 80, 120)
DROPOUT_RATE: float = 0.2

# Scaler per prediction target: closing prices are bounded, so MinMax suits them;
# log returns are stationary, so a standard scaler is the natural choice.
SCALER_KIND_BY_TARGET: dict[str, str] = {
    "price": "minmax",
    "price_change": "minmax",
    "log_return": "standard",
}

# Three targets, deliberately:
#
#   "price"        the original tutorial's task, kept exactly as specified. Under a
#                  train-only MinMax scaler the test period trades far above anything
#                  the scaler was fitted on, so this target is expected to saturate.
#                  It is kept as the measured cost of the level parameterisation.
#   "price_change" the same one-step-ahead price forecast, but the model predicts the
#                  next-day *change* in dollars and the price is rebuilt as
#                  previous + change. Changes oscillate around zero instead of trending,
#                  so the train-fitted scaler covers 99.1% of the test period; the four
#                  days that escape it are 2021-2022 outlier rallies, not drift. This is
#                  the price forecast that actually works, and the pipeline reports both
#                  how many days leave the fitted range and by how much.
#   "log_return"   the honest stationary target: the model cannot score well by
#                  copying yesterday's price.
PREDICTION_TARGETS: tuple[str, ...] = ("price", "price_change", "log_return")

PRICE_TARGET: str = "price"
PRICE_CHANGE_TARGET: str = "price_change"
LOG_RETURN_TARGET: str = "log_return"

# Whether the *previous* value of the target predicts its next value. This decides how
# ``corr(window[-1], target)`` should be read, and the two cases are opposites:
#
#   persistent (the price level)   the last price in a window almost perfectly predicts
#                                  the next one. A correlation near +1 is required, and a
#                                  value near zero is the signature of the original
#                                  notebook's shuffled windows (-0.0248).
#   stationary (change, log return) the target is ~white noise, so the previous value
#                                  predicts the next one at roughly zero correlation *by
#                                  construction*. A value near zero here means the target
#                                  behaves as expected and says nothing about ordering,
#                                  which is exactly why the shuffle guard cannot be
#                                  applied to it.
#
# Without this distinction the pipeline prints one number with two opposite readings and
# leaves the reader to guess which one applies.
PERSISTENT_TARGETS: frozenset[str] = frozenset({PRICE_TARGET})


def target_is_persistent(prediction_target: str) -> bool:
    """True when the previous target value is expected to predict the next one."""
    return prediction_target in PERSISTENT_TARGETS


def model_path_for(prediction_target: str) -> Path:
    """Checkpoint filename for a given prediction target."""
    return MODEL_DIR / f"lstm_googl_{prediction_target}.keras"


def ensure_directories() -> None:
    """Create the output directories if they do not exist yet."""
    for directory in (DATA_DIR, MODEL_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
