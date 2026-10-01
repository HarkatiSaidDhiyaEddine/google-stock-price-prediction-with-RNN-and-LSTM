"""Chronological LSTM pipeline for GOOGL - fixes the constant-prediction bug.

The only substantive change from the notebook is that the train/test split is
chronological instead of a shuffled `train_test_split`. The scaler is fitted on
the whole price series (min/max only, as in the original tutorial) so that the
later, higher-priced test period is not compressed.
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "GOOGL.csv"
PLOT_PATH = PROJECT_DIR / "prediction_vs_actual.png"

TARGET_COL = "Open"      # the notebook trains on Open; change to "Close" for the closing price
WINDOW = 60              # timesteps per input sequence
TEST_FRACTION = 0.10     # last 10% of the timeline is the test period
EPOCHS = 5               # the notebook uses 100; 5 is enough to see real tracking
BATCH_SIZE = 32
SEED = 42


def load_prices(path, column):
    """Read the CSV and return it sorted by date, oldest first."""
    frame = pd.read_csv(path)
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.sort_values("Date").reset_index(drop=True)
    return frame[["Date", column]]


def make_windows(scaled_values, window):
    """Turn a 1-D scaled series into (samples, window, 1) inputs and 1-D targets."""
    inputs = np.stack([scaled_values[i - window:i] for i in range(window, len(scaled_values))])
    targets = scaled_values[window:]
    return inputs[..., np.newaxis], targets


def build_model(window):
    """The notebook's exact 4-layer LSTM stack."""
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(60, activation="relu", return_sequences=True, input_shape=(window, 1)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(60, activation="relu", return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(80, activation="relu", return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(120, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    prices = load_prices(DATA_PATH, TARGET_COL)
    values = prices[TARGET_COL].to_numpy(dtype=float)

    split_index = len(prices) - int(len(prices) * TEST_FRACTION)
    train_values = values[:split_index]
    test_values = values[split_index:]
    test_dates = prices["Date"].iloc[split_index:].to_numpy()

    print("timeline          : %s -> %s (%d rows)" % (prices["Date"].iloc[0].date(),
                                                      prices["Date"].iloc[-1].date(), len(prices)))
    print("train period      : %s -> %s (%d rows)" % (prices["Date"].iloc[0].date(),
                                                      prices["Date"].iloc[split_index - 1].date(),
                                                      len(train_values)))
    print("test period       : %s -> %s (%d rows)" % (prices["Date"].iloc[split_index].date(),
                                                      prices["Date"].iloc[-1].date(),
                                                      len(test_values)))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_all = scaler.fit_transform(values.reshape(-1, 1))[:, 0]
    train_scaled = scaled_all[:split_index]
    test_scaled = scaled_all[split_index:]

    x_train, y_train = make_windows(train_scaled, WINDOW)
    print("x_train shape     :", x_train.shape, " y_train shape:", y_train.shape)
    print("corr(window[-1], target) = %+.4f" % np.corrcoef(x_train[:, -1, 0], y_train)[0, 1])

    model = build_model(WINDOW)
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2)
    print("loss per epoch    :", [round(v, 6) for v in history.history["loss"]])

    # Test inputs: the last WINDOW real training days, then the test period itself.
    test_sequence = np.concatenate([train_scaled[-WINDOW:], test_scaled])
    x_test, _ = make_windows(test_sequence, WINDOW)

    predicted = scaler.inverse_transform(model.predict(x_test, verbose=0)).ravel()
    actual = test_values

    rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))
    mae = float(np.mean(np.abs(predicted - actual)))
    naive_rmse = float(np.sqrt(np.mean((actual[1:] - actual[:-1]) ** 2)))
    correlation = float(np.corrcoef(predicted, actual)[0, 1])

    print("\npredicted std=%.2f  min=%.2f  max=%.2f" % (predicted.std(), predicted.min(), predicted.max()))
    print("actual    std=%.2f  min=%.2f  max=%.2f" % (actual.std(), actual.min(), actual.max()))
    print("RMSE=%.2f  MAE=%.2f  corr(pred, actual)=%+.4f  (naive last-value RMSE=%.2f)"
          % (rmse, mae, correlation, naive_rmse))

    print("\nfirst 10 test days (predicted vs actual):")
    for date, pred, truth in list(zip(test_dates, predicted, actual))[:10]:
        print("  %s  predicted %9.2f  actual %9.2f" % (pd.Timestamp(date).date(), pred, truth))
    print("last 5 test days (predicted vs actual):")
    for date, pred, truth in list(zip(test_dates, predicted, actual))[-5:]:
        print("  %s  predicted %9.2f  actual %9.2f" % (pd.Timestamp(date).date(), pred, truth))

    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, actual, label="actual %s" % TARGET_COL)
    plt.plot(test_dates, predicted, label="predicted %s" % TARGET_COL)
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.title("Chronological split - %s, one-step-ahead prediction" % TARGET_COL)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=120)
    print("\nsaved plot to", PLOT_PATH)


if __name__ == "__main__":
    main()
