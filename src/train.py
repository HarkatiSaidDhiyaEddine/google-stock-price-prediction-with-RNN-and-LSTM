"""Training loop with a chronological validation split and early stopping.

Two things matter here:

1. The validation set is the **tail of the training period**, not a random sample.
   A random validation split would leak future prices into the epoch-by-epoch
   decisions that early stopping makes, which is the same class of mistake as the
   original shuffled train/test split.
2. Training stops when validation loss stops improving. The original notebook ran a
   fixed 100 epochs on a CPU-only machine with no way to know when to stop.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.model import build_lstm, compile_model


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def chronological_validation_split(
    inputs: np.ndarray, targets: np.ndarray, validation_fraction: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split samples by position: the newest ``validation_fraction`` become validation."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(f"validation_fraction must be in (0, 1), got {validation_fraction}")

    validation_count = int(len(inputs) * validation_fraction)
    if validation_count <= 0 or validation_count >= len(inputs):
        raise ValueError(
            f"validation_fraction={validation_fraction} leaves {validation_count} "
            f"validation samples out of {len(inputs)}"
        )

    boundary = len(inputs) - validation_count
    return inputs[:boundary], targets[:boundary], inputs[boundary:], targets[boundary:]


def train_model(
    inputs: np.ndarray,
    targets: np.ndarray,
    model_path: Path,
    validation_fraction: float,
    epochs: int,
    batch_size: int,
    patience: int,
    cell_activation: str,
    hidden_units: tuple[int, ...],
    dropout: float,
    window: int,
    verbose: int = 2,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History, tuple[np.ndarray, np.ndarray]]:
    """Build, fit and checkpoint the LSTM.

    Returns:
        The trained model, its training history, and the ``(inputs, targets)``
        validation slice so the caller can report validation loss.
    """
    x_fit, y_fit, x_val, y_val = chronological_validation_split(
        inputs, targets, validation_fraction
    )

    model = build_lstm(
        window=window,
        units=hidden_units,
        dropout=dropout,
        cell_activation=cell_activation,
    )
    compile_model(model)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    history = model.fit(
        x_fit,
        y_fit,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose,
    )
    return model, history, (x_val, y_val)


def load_trained_model(model_path: Path) -> tf.keras.Model:
    """Load a checkpoint saved by :func:`train_model`."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"no checkpoint at {model_path}. Run the pipeline first."
        )
    return tf.keras.models.load_model(model_path)
