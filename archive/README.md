# Archive — historical artifacts, not part of the pipeline

Nothing in this directory is imported, read, or executed by `src/`, `tests/`, or
`run_pipeline.py`. The working project is the package in `../src/`, driven by
`../run_pipeline.py`. Everything here is kept because it is either the *evidence*
for the diagnosis in `../README.md` or the *predecessor* of the current code.

If you are looking for the project, stop here and read `../README.md` instead.

---

## `train.csv` / `test.csv` — do not use these as data

These are the artifacts of the original notebook's
`train_test_split(df, test_size=0.1, random_state=42)` call. They are kept only
because they are the thing the project's central claim was measured on, and
re-verifying that claim is cheap with them present.

Two properties make them unusable and unrecoverable:

1. **They are shuffled.** The rows are in random order, so 60-row windows built
   from them contain 60 unrelated trading days rather than 60 consecutive ones.
   The first eight `Open` values in `train.csv` are
   `2263.57, 1152.28, 154.79, 58.76, 2719.51, 116.92, 323.84, 1055.02` — days
   from across 2004–2022, in no order.
2. **They have no `Date` column.** The split cell called `set_index("Date")` and
   then wrote with `index=False`, so the timeline is gone and the original order
   cannot be recovered.

Together these produce the original bug: `corr(window[-1], target) = -0.0248`, so
the MSE-optimal prediction is the constant mean (≈ $595), and the notebook duly
predicted ≈ $595 for every day.

They can be regenerated from `../data/GOOGL.csv` at any time — with a
*chronological* split, which is what `../src/preprocessing.py` does. To reproduce
the measurements above:

```python
import pandas as pd
frame = pd.read_csv("archive/train.csv")
print(frame["Open"].head(8).tolist())
```

**Never pass these files to the model.** They are here to be measured, not trained on.

---

## `Untitled_original_backup.ipynb`

The original notebook, preserved unmodified: EDA, the shuffled split, the dead
`MA_for_251_days` cell that raises `KeyError`, the `scaled_data` cell that raises
`NameError`, and the 4-layer LSTM with `activation='relu'`.

It is byte-identical to the `Untitled.ipynb` that was deleted from the project
root — the two files hashed to the same SHA-256
(`18C31ec0…0eda9e`), so only one copy was kept. It is superseded by
`../notebooks/01_eda.ipynb` and `../notebooks/02_lstm_walkthrough.ipynb`, which
import `src/` and contain no model logic of their own.

---

## `lstm_chronological_fix.py`

The standalone corrected pipeline that came before the package. It proved that a
chronological split cures the constant output:

```
corr(window[-1], target) = +0.9996   loss 0.00372 → 0.000483
corr(pred, actual)       = +0.9845
```

It also showed the limit of that fix: **RMSE 403.74 against a naive persistence
baseline of 41.42**. Un-shuffling produces a *working* project, not a *good*
result — which is why the current pipeline reports every metric against the
baseline and adds the stationary targets (`price_change`, `log_return`).

It is superseded by `../src/run_pipeline.py`. One methodological difference is
deliberate: it fitted its `MinMaxScaler` on the whole series (mild leakage of
future min/max), whereas the current pipeline fits on the training slice only.

---

## `prediction_vs_actual.png`

The plot written by `lstm_chronological_fix.py`. Superseded by the figures in
`../figures/`, one set per prediction target.
