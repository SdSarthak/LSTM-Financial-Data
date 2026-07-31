# LSTM Financial Data

LSTM forecasting for financial and macroeconomic time series, built around three
public datasets: daily world stock prices, a wide World Bank/IMF style indicator
export, and the Reserve Bank of India's *Select Economic Indicators* workbook.

The point of the project is a pipeline you can trust: chronological splits,
scalers fitted on training rows only, and every model scored against a naive
persistence baseline so the numbers mean something.

## What it does

- **Loaders** for all three source files, including the awkward parts: mixed
  timezone offsets and duplicated daily snapshots in the stock CSV, `..` missing
  markers and `2023M04 [2023M04]` period columns in the indicator export, and
  merged multi-row headers with sparse fiscal-year labels in the RBI workbook.
- **Technical indicators**: log returns, SMA, EMA, RSI (Wilder), MACD with
  signal and histogram, Bollinger bands with %B, annualised realised volatility.
- **Leak-free supervised windows**: the split is chronological, `MinMaxScaler`
  sees only training rows, and windows straddling a split boundary are dropped.
- **Stacked LSTM** with dropout, early stopping and LR reduction on plateau,
  saved in the native `.keras` format together with a JSON run summary.
- **Two modelling targets** - next log return (default) or price level - with
  all metrics reported in price units so the two are comparable.
- **Honest evaluation**: RMSE, MAE, MAPE, R2 and directional accuracy for the
  LSTM *and* for a "tomorrow = today" persistence baseline, side by side.
- **Charts**: predictions vs actual, loss curves, and the recursive forecast.

## Install

```bash
git clone https://github.com/SdSarthak/LSTM-Financial-Data.git
cd LSTM-Financial-Data
python -m venv venv
venv\Scripts\activate          # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env           # then edit LSTM_FIN_DATA_DIR
```

TensorFlow is only needed for training. `explore` and `indicators` work without
it, and the test suite skips the model tests when it is missing.

## Getting the data

No dataset is committed to this repository - the files are large and not mine to
redistribute. Download them into the directory you set as `LSTM_FIN_DATA_DIR`:

| Dataset | Source | Default file name |
| --- | --- | --- |
| World stock prices (daily OHLCV, ~50 MB) | Kaggle: *World Stock Prices - Daily Updating* by nelgiriyewithana | `World-Stock-Prices-Dataset.csv` |
| Economic indicators (wide series x period export) | World Bank DataBank / IMF IFS, exported as CSV with the "series x time" layout | `<export-id>_Data.csv` |
| RBI Select Economic Indicators | Reserve Bank of India, *Handbook of Statistics* - Table No. 01 | `RBIB Table No. 01 _ Select Economic Indicators.xlsx` |

Point `.env` at whatever you named them:

```env
LSTM_FIN_DATA_DIR=/path/to/data
LSTM_FIN_STOCK_FILE=World-Stock-Prices-Dataset.csv
LSTM_FIN_ECONOMIC_FILE=my-export_Data.csv
LSTM_FIN_RBI_FILE=RBIB Table No. 01 _ Select Economic Indicators.xlsx
```

You do not need any of them to try the project: `demo` trains on a synthetic
series generated in-process.

## Usage

```bash
python main.py                                   # synthetic demo, no data needed
python main.py demo --epochs 40 --lookback 30

python main.py explore stocks                    # list tickers
python main.py explore economic --limit 10       # series coverage
python main.py explore rbi --sheet Monthly       # indicators in the workbook

python main.py indicators --ticker AAPL --plot   # export technical indicators
python main.py train --ticker AAPL --lookback 60 --epochs 30
python main.py economic --series-code CORENS --country GBR --frequency M --lookback 6
```

`python -m lstm_financial <command>` and the `lstm-financial` console script
(after `pip install -e .`) do exactly the same thing.

Useful flags for the training commands:

| Flag | Meaning |
| --- | --- |
| `--target-mode return\|level` | predict the next log return (default) or the price itself |
| `--features log_return rsi_14 ...` | choose the input columns; a single feature enables the recursive forecast |
| `--horizon 5` | direct multi-step forecasting: predict 5 steps per window |
| `--forecast-steps 30` | length of the recursive forecast (univariate models only) |
| `--units 64 32 --dropout 0.2` | architecture |
| `--test-fraction 0.2 --validation-fraction 0.1` | chronological split sizes |
| `--artifacts-dir ./artifacts` | where plots, the `.keras` model and the JSON summary go |

## Library use

```python
from lstm_financial import datasets, run_forecast

history = datasets.load_ticker_history("AAPL")        # date-indexed OHLCV
run = run_forecast(history, name="aapl", lookback=60, epochs=30)

print(run.report())
print(run.metrics["RMSE"], run.baseline_metrics["RMSE"], run.beats_baseline())
```

Lower-level pieces are importable on their own: `add_technical_indicators`,
`build_dataset`, `FinancialLSTM`, `regression_metrics`.

## Why log returns are the default

Prices trend. A scaler fitted on 2000-2020 maps 2021-2025 prices far outside
`[0, 1]`, and an LSTM cannot extrapolate past the range it was trained on. On
AAPL (6,408 daily rows, 1,216 test windows, 30 epochs) the same architecture
scored:

| Target mode | RMSE | MAE | MAPE | R2 |
| --- | --- | --- | --- | --- |
| `level` (price) | 96.90 | 91.19 | 52.25% | -6.37 |
| `return` (default) | 11.06 | 7.32 | 4.22% | 0.90 |
| persistence baseline | 3.10 | 2.20 | 1.34% | 0.99 |

Note the last row. On daily equity prices the naive "tomorrow = today"
forecast still wins, which is the expected result and the reason the baseline is
printed on every run. Treat `beats_baseline()` as the bar to clear, not the
R2 of the price curve - a high R2 against a trending series mostly measures the
trend. Directional accuracy is the metric worth optimising, and at ~47% this
model is not yet finding a signal.

## Project layout

```
lstm_financial/
  config.py      environment-driven paths and .env loading
  datasets.py    loaders for the three datasets + synthetic generators
  features.py    technical indicators
  windows.py     scaling, chronological splitting, sequence building
  model.py       FinancialLSTM (TensorFlow imported lazily)
  metrics.py     RMSE/MAE/MAPE/R2/directional accuracy
  plots.py       chart helpers (headless by default)
  pipeline.py    end-to-end run + persistence baseline + artifacts
  cli.py         demo / train / economic / explore / indicators
tests/           deterministic tests on synthetic fixtures only
main.py          thin entry point around the CLI
```

## Tests

```bash
pytest
```

102 tests, all on synthetic fixtures: no network, no database, and none of the
real datasets are read. The TensorFlow-dependent tests train two-epoch models on
a few hundred windows and are skipped if TensorFlow is not installed.

## Notes and limitations

- Recursive multi-step forecasting is only defined for univariate models: with
  exogenous features there is nothing to feed the next step. Use `--horizon` for
  direct multi-step forecasting with the full feature set instead.
- The economic export mixes annual, quarterly and monthly columns; pick one
  frequency with `--frequency` before modelling.
- RBI quarters are fiscal (`2024-25 Q1` starts in April 2024) and the workbook is
  newest-first; the loader converts both.
- Hyper-parameters are not tuned. No walk-forward or purged cross-validation
  yet - the split is a single chronological train/validation/test.

## Disclaimer

Educational and research code. Nothing here is investment advice.

## License

MIT
