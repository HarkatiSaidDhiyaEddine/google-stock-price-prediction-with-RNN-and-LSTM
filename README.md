# GOOGL next-day price prediction with a stacked LSTM

Next-day forecasting of the Alphabet (GOOGL) closing price from `data/GOOGL.csv`
(4,431 trading days, 2004-08-19 → 2022-03-24), using the four-layer LSTM 
---

## 3. How to run

```bash
pip install -r requirements.txt

python run_pipeline.py          # trains all three targets, writes figures + checkpoints
python -m pytest tests/ -v      # 50 tests, ~3 s
```

Useful environment variables:

```bash
LSTM_EPOCHS=3 python run_pipeline.py    # smoke run instead of the 100-epoch ceiling
```

`EPOCHS` is a ceiling, not a schedule: `EarlyStopping(patience=10)` stops training
when validation loss stops improving, so a full run trains until it stops helping.
On this CPU-only machine the three targets took about 45 minutes together.

### Or in Jupyter

All three notebooks are first-class entry points. They import the same `src/` package
the command line uses and hold no model logic of their own, and Jupyter is already a
dependency (verified with `jupyterlab` 3.6.3 / `notebook` 6.5.4 / `nbconvert` 6.5.4):

```bash
python -m jupyter lab          # or: python -m jupyter notebook
```

Launch it from the project root **or** from `notebooks/` — either works. The setup cell
puts the project root on `sys.path`, and `src/config.py` derives every path from
`Path(__file__)`, so `data/`, `figures/` and `models/` resolve no matter where the
kernel starts.

| notebook | what it does | runtime |
|---|---|---|
| `notebooks/01_eda.ipynb` | describes `data/GOOGL.csv`, writes the EDA figures. No model. | seconds |
| `notebooks/02_lstm_walkthrough.ipynb` | the price target through split → scale → window → model → fit → forecast → evaluate → plot | minutes — it trains |
| `notebooks/03_full_pipeline.ipynb` | **the whole project from one file**: runs all three targets through `src.run_pipeline.main()`, renders every figure inline, and executes the test suite | as long as `python run_pipeline.py` — it trains all three |

`03_full_pipeline.ipynb` is the single-file entry point: open it, run all, and the
project has run end to end — pre-flight checks on the environment and the data, all
three targets trained and scored beside the naive baseline, the summary table as a
frame, the sample forecasts, the four EDA figures, the nine per-target figures, the
50-test suite, and a final listing of what was written to `figures/` and `models/`.
Everything it calls is the same `src/` code the command line runs, so a number produced
in the notebook and a number produced by `python run_pipeline.py` come from one
implementation.

Because 02 and 03 train, cap the epoch ceiling while you explore:

```cmd
set LSTM_EPOCHS=3 && python -m jupyter lab
```

```powershell
$env:LSTM_EPOCHS="3" ; python -m jupyter lab
```

The whole pipeline is callable from a cell as well, which is the fastest way to iterate
without the walkthrough's step-by-step ceremony:

```python
%run run_pipeline.py                    # same as `python run_pipeline.py`
```

```python
from src.run_pipeline import main       # or drive it directly
runs = main()                           # one TargetRun per prediction target
```

Individual stages are importable too, so a cell can re-run just one of them:
`run_price_target`, `run_price_change_target` and `run_log_return_target` each take
`(prices, split, verbose)` and return a `TargetRun`.

Headless execution is how this repository's own verification works, and how all three
notebooks' stored outputs were produced — a fresh kernel, start to finish. The
`--ExecutePreprocessor.timeout=-1` on the notebook that trains is required: nbconvert's
default is a 30-second limit per cell, and a training cell runs far longer.

```bash
# the whole project, all three targets, from one file
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/03_full_pipeline.ipynb

python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/02_lstm_walkthrough.ipynb
```

---

## 4. Results

Setup for every number below: chronological 90/10 split (3,987 train / 444 test rows,
test period 2020-06-22 → 2022-03-24), 60-day windows, four LSTM layers of
60/60/80/120 units with `tanh` and dropout 0.2, Adam + MSE, batch 32, seed 42,
`EarlyStopping(patience=10)` on a chronological validation tail. The scaler is fitted
on the training slice only.

Every forecast is one day ahead and uses only prices known up to the day before the
forecast: prediction for day `t+1` is built from days `t-59 … t`.

**The bar to clear:** of the 444 test days, 248 closed above the previous close, so an
idle forecast that always answers "up" scores a directional accuracy of **0.5586**. The
naive persistence forecast ("tomorrow = today") scores **RMSE $38.65**.

| target | network output std | actual std | corr(pred, actual) | RMSE model | RMSE naive | MAE model | MAE naive | R² model | R² naive | dir. acc. | up-days | beats RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `price` | $66.54 | $520.30 | +0.6420 | 826.73 | **38.65** | 673.68 | **27.94** | −1.5248 | **0.9945** | 0.4505 | 0.5586 | no |
| `price_change` | $520.94 | $520.30 | +0.9973 | 38.6779 | **38.6486** | **27.9325** | 27.9429 | 0.9945 | 0.9945 | 0.5023 | 0.5586 | no |
| `log_return` | $521.44 | $520.30 | +0.9973 | **38.5516** | 38.6486 | **27.7622** | 27.9429 | 0.9945 | 0.9945 | 0.5586 | 0.5586 | by 0.097 (0.25%) |

Read the last three columns together — they are the whole story:

### `price` — the level target fails, and the reason is structural

The predicted price standard deviation is $66.54 against an actual $520.30: the model
tracks the shape of the curve but sits far too flat. RMSE is **21× worse than
persistence**. The cause is measurable rather than mysterious: the scaler is fitted on
the training slice only, and **87.4%** of the test period trades outside the range it
ever saw — a mean overshoot of **0.49** and a maximum of **0.998** in units of that
range, i.e. the test period reaches roughly double the largest training price. A
`MinMaxScaler` saturates under that condition, so the forecast is compressed toward the
training ceiling. This is the honest cost of the tutorial's original parameterisation,
kept in the pipeline precisely so it can be measured.

### `price_change` — difference encoding fixes extrapolation, not the forecast

Encoding the target as a price *difference* fixes the extrapolation problem: only
**0.9%** of test days (four 2021-2022 outlier rallies) leave the fitted range, with a
maximum overshoot of 0.389 and a mean of 0.0016. The predicted price spread,
$520.94, is now essentially the actual one, and `corr(pred, actual) = +0.9973`.

That correlation is **not** skill. The price is rebuilt as `previous_close + predicted
change`, so it inherits almost all of its variance from the previous close — the exact
information the naive baseline is handed for free. The network's *own* output standard
deviation is 1.14 against an actual change spread of 38.52: a ratio of **0.0296**. It
has learned to emit an almost constant change, which is why its directional accuracy
(0.5023) is *below* the 0.5586 up-day base rate, and why its RMSE loses to persistence
by 0.029 — 0.08%.

### `log_return` — one hairline win, and no directional skill

The log-return target beats persistence on RMSE by **0.097, or 0.25%**, and on MAE by
0.181. Its directional accuracy is **0.5586 — exactly the up-day share**. A forecast
that ignores the input entirely and always says "up" scores precisely the same number.
The network's own output standard deviation is 0.000115 against an actual return spread
of 0.017256, a ratio of **0.0067**: it is emitting the average daily drift, and the
RMSE edge exists because the test period drifted upward at roughly that rate.

The pipeline prints this as a diagnosis rather than a headline:

```
diagnosis: network output effectively constant in target space
  price_change  output std 1.14196 vs actual target std 38.5185   (ratio 0.0296, below 5%)
  log_return    output std 0.000115297 vs actual target std 0.0172557  (ratio 0.0067, below 5%)
```

**Summary: the LSTM does not beat "tomorrow's price = today's price."** Persistence
already reaches R² 0.9945 because prices are extremely autocorrelated. The only
apparent win is 0.25% of RMSE on the log-return target, with directional accuracy
sitting exactly on the base rate and a network output that is nearly constant — which
is not a result to build a trading system on.

## 5. Why the model behaves this way

Two facts about this data explain every number above:

1. **A daily close is almost entirely predictable from yesterday's close.** That is
   what the R² 0.9945 baseline is measuring. Anything a model has to add is a fraction
   of a percent, and it must add it *without* the anchor.
2. **Daily log returns are close to unpredictable.** Once the level is removed, the
   remaining signal is small relative to the noise, so the MSE-optimal forecast is very
   near the mean return. An LSTM trained with MSE therefore converges to an almost
   constant output — which is why `price_change` and `log_return` show target-space
   variance ratios of 0.03 and 0.007.

This is why every metric in this repository is printed next to the baseline, and why
the pipeline refuses to report the difference targets as if their reconstructed price
curves were forecasts.

## 6. Limitations — read before quoting any number above

- **One ticker, one horizon, one split.** GOOGL, one day ahead, a single 90/10
  chronological split. No walk-forward or rolling-origin evaluation, so the
  log-return target's 0.25% RMSE edge is not shown to hold out of sample.
- **That 0.25% edge is not demonstrated to be more than noise.** There is no
  significance test, no confidence interval, and no repeated-seed study on it.
  Given that the same model's directional accuracy lands exactly on the up-day base
  rate, the honest reading is "indistinguishable from persistence".
- **No hyperparameter search.** The architecture is the tutorial's, deliberately, so
  the comparison is like-for-like. The one documented change is `tanh` vs `relu` on
  the LSTM cell, kept behind `config.CELL_ACTIVATION` so it can be measured rather
  than argued about.
- **Loss is MSE.** Nothing was trained to optimise direction; directional accuracy is
  reported as an outcome, not an objective.
- **Not tradable.** No transaction costs, no slippage, no position sizing, no
  multi-day holding. Close-to-close forecasting with same-day information is a
  learning exercise, not a strategy.
- **`price_change` and `log_return` are not extrapolation-free.** Four 2021-2022
  outlier days overshoot the training range for `price_change` (0.9% of days, maximum
  overshoot 0.389 of the range). The pipeline reports this instead of clipping it.
- **CPU-only TensorFlow** with oneDNN enabled, which the library itself warns can
  shift results at the last decimal places between runs. Numbers here are from one
  run at seed 42.
- **The test period is a bull market** (2020-06 → 2022-03), so persistence and any
  drift-following forecast both look good in absolute terms. The up-day share of
  0.5586 is a property of this window, not of GOOGL forever.

## 7. Repository layout

```
google stock price prediction with RNN and LSTM/
├── data/GOOGL.csv              single source of truth (4,431 rows, 2004-2022)
├── src/
│   ├── config.py               paths, seeds, hyper-parameters, target definitions
│   ├── data_loader.py          load + validate + sort; raises on nulls/dupes
│   ├── preprocessing.py        chronological split, per-target scalers, windowing
│   ├── model.py                the 4-layer LSTM
│   ├── train.py                chronological validation + early stopping + checkpoint
│   ├── evaluate.py             RMSE/MAE/MAPE/R², direction accuracy, base rate
│   ├── plots.py                every figure, headless (Agg)
│   └── run_pipeline.py         end-to-end entry point; prints the metrics tables
├── notebooks/
│   ├── 01_eda.ipynb            data description only, no model logic
│   ├── 02_lstm_walkthrough.ipynb  imports src/, one stage per cell
│   └── 03_full_pipeline.ipynb  the whole project, all three targets, one file
├── tests/                      50 tests
├── conftest.py                 puts the project root on sys.path for pytest
├── figures/                    generated PNGs (gitignored)
├── models/                     .keras checkpoints (gitignored)
├── archive/                    the original notebook, the shuffled CSVs, the evidence
├── plan.md                     the remake plan and its locked design decisions
├── project_info__1.md          the original codebase review and root-cause analysis
├── project_info__2.md          what the original notebook needed vs what it had
└── README.md                   this file
```

`archive/` holds the evidence for section 1 and is imported by nothing. In
particular `archive/train.csv` and `archive/test.csv` are the shuffled,
`Date`-stripped split artifacts of the original notebook: they are kept to be
*measured*, never trained on — they cannot even be repaired in place, because the
timeline was dropped when they were written.

## 8. Verification

```bash
python run_pipeline.py          # exits 0, prints the three per-target tables + summary
python -m pytest tests/ -v      # 50 passed
```

The same two checks are reproducible from a single Jupyter file, which is also how this
repository's own end-to-end verification is run:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/03_full_pipeline.ipynb
```

What the tests actually assert, beyond ordinary unit coverage:

| Test | Guarantee |
|---|---|
| `test_window_last_value_correlates_with_target` | `corr(window[-1], target) > 0.9` on the real data. This is the check that would have caught the original bug immediately; it fails loudly if a shuffle is ever reintroduced. |
| `test_windows_use_consecutive_trading_days` | the 60 rows in a window are consecutive trading days (no gap > 7 calendar days) |
| `test_split_is_positional_and_contiguous` | 3,987 / 444 rows, train ends exactly where test begins, no overlap |
| `test_no_shuffling_calls_exist_in_the_source_tree` | AST scan of `src/` for `train_test_split`, `shuffle`, `permutation`, `sample` |
| `test_no_shuffle_keyword_arguments_are_used` | AST scan for a `shuffle=` keyword argument anywhere in `src/` |
| `test_scaler_is_fitted_on_train_only_and_reports_extrapolation` | the scaler never saw a test price |
| `test_price_change_target_needs_far_less_extrapolation_than_the_price_level` | the 0.9% vs 87.4% extrapolation contrast is a regression test, not a claim in prose |
| `test_price_change_overshoot_is_confined_to_outliers` | ...and the overshoot *magnitude* stays small, which a day count alone cannot express |
| `test_up_day_share_equals_what_an_always_up_forecast_scores` | direction accuracy and the base rate are the same measurement, so printing one without the other is not allowed |
| `test_only_the_price_level_is_persistent` | the shuffle guard is applied to the price level and not to a stationary target, where the correlation is ~0 by construction |

The pipeline also refuses to report a run in which a predicted *price* level
collapses to a constant (`_assert_price_predictions_vary`, threshold $50 of standard
deviation), and prints a diagnosis — rather than a headline — when a target is
predicted as an effectively constant value in its own units.


