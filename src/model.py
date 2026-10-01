"""The stacked LSTM, matching the original notebook's topology.

The notebook used four LSTM layers of 60/60/80/120 units with ``Dropout(0.2)``
between them and a single ``Dense(1)`` output. That topology is kept here.

One detail is corrected. Keras' LSTM has *two* activation functions:

* ``activation``          -- the cell *output* activation, default ``tanh``
* ``recurrent_activation`` -- the *gate* activation, default ``sigmoid``

The notebook passed ``activation='relu'``, so it replaced the ``tanh`` cell output
with an unbounded ``relu``. That is what this module makes configurable, via
``cell_activation``. The gate activation is left at its default.
"""
from __future__ import annotations

import tensorflow as tf

DEFAULT_LEARNING_RATE = 1e-3


def build_lstm(
    window: int,
    units: tuple[int, ...],
    dropout: float,
    cell_activation: str,
    output_units: int = 1,
    feature_count: int = 1,
) -> tf.keras.Model:
    """Build the stacked LSTM.

    Args:
        window: number of timesteps per input sequence.
        units: hidden size of each LSTM layer, in order.
        dropout: dropout rate applied after every LSTM layer. ``0`` disables it.
        cell_activation: ``"tanh"`` (Keras default) or ``"relu"`` (the notebook's
            choice, kept selectable so the difference is measurable).
        output_units: size of the final ``Dense`` layer.
        feature_count: number of input features per timestep.

    Returns:
        An uncompiled :class:`tf.keras.Model`.
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if not units:
        raise ValueError("at least one LSTM layer is required")
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {dropout}")

    model = tf.keras.Sequential(name="googl_lstm")
    model.add(tf.keras.layers.Input(shape=(window, feature_count), name="price_window"))

    for index, unit_count in enumerate(units):
        is_final_layer = index == len(units) - 1
        model.add(
            tf.keras.layers.LSTM(
                units=unit_count,
                activation=cell_activation,
                return_sequences=not is_final_layer,
                name=f"lstm_{index + 1}_{unit_count}",
            )
        )
        if dropout > 0.0:
            model.add(tf.keras.layers.Dropout(dropout, name=f"dropout_{index + 1}"))

    model.add(tf.keras.layers.Dense(units=output_units, name="output"))
    return model


def compile_model(model: tf.keras.Model, learning_rate: float = DEFAULT_LEARNING_RATE) -> tf.keras.Model:
    """Compile with Adam and mean squared error, as in the original notebook."""
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mean_squared_error")
    return model
