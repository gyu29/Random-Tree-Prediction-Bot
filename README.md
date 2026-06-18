# Random Tree Prediction Bot

A Python swing-trading research system that trains an ensemble classifier on historical market data, detects possible swing opportunities, and provides a modern PySide6 desktop terminal for monitoring model settings, watchlists, alerts, analysis, training, and backtest-style summaries.

> This project is for education and research only. It is not financial advice, and it should not be used as the sole basis for real trading decisions.

## Features

- Trains a swing-trading classifier from local historical OHLCV CSV files.
- Uses a hybrid Random Forest + XGBoost ensemble when `xgboost` is installed.
- Builds technical-analysis features with `ta`, `pandas`, `numpy`, and scikit-learn.
- Saves reusable model artifacts with `joblib`.
- Supports Korean market lookups through data.go.kr / KRX endpoints.
- Supports optional US market lookups through Alpha Vantage.
- Includes a native PySide6 desktop UI for training, analysis, monitoring, screening, settings, and backtesting.
- Uses cached `QStackedWidget` pages so navigation updates existing widgets instead of rebuilding screens.
- Uses PyQtGraph for responsive price and equity-curve visualization.
- Uses local historical CSV fallbacks so the UI remains useful when provider API keys are unavailable.
- Includes PyInstaller specs and build commands for packaging the Python application as a standalone executable.

## Project Layout

```text
Random-Tree-Prediction-Bot/
|-- historical_data/                 # Local CSV training data
|-- swing_trading_system.py          # Trading-system logic and application entry point
|-- qt_trading_ui.py                 # Cached PySide6 UI and PyQtGraph charts
|-- download_sp500_data.py           # Downloads sample historical data with yfinance
|-- key_tester.py                    # Tests data.go.kr KRX service-key access
|-- stocks.json                      # Symbol list used by the stock screener and watchlist
|-- swing_model_enhanced.pkl         # Saved trained model
|-- swing_scaler_enhanced.pkl        # Saved feature scaler
|-- feature_columns_enhanced.pkl     # Saved model feature list
|-- training_stats.pkl               # Saved model training metadata
|-- .env.example                     # Example runtime secrets file
|-- *.spec                           # PyInstaller build specs
`-- pyinstaller_cmd.txt              # Manual PyInstaller command examples
```

## Requirements

### Python

- Python 3.10 or newer recommended
- `pip`

Install the Python packages used by the scripts:

```bash
pip install pandas numpy scikit-learn ta joblib requests xgboost yfinance PySide6 pyqtgraph pyinstaller
```

`xgboost` is required for training the current hybrid ensemble. `yfinance` is only needed when running `download_sp500_data.py`. `pyinstaller` is only needed when building an executable.

## Configuration

Copy the example environment file:

```bash
copy .env.example .env
```

On macOS/Linux:

```bash
cp .env.example .env
```

Then edit `.env`:

```env
KRX_SERVICE_KEY=your_data_go_kr_service_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
```

`KRX_SERVICE_KEY` is used for Korean market data. `ALPHA_VANTAGE_API_KEY` is optional and only needed for live US market lookups.

You can also set secrets directly in the shell.

PowerShell:

```powershell
$env:KRX_SERVICE_KEY="your_data_go_kr_service_key"
$env:ALPHA_VANTAGE_API_KEY="your_alpha_vantage_key"
```

macOS/Linux:

```bash
export KRX_SERVICE_KEY="your_data_go_kr_service_key"
export ALPHA_VANTAGE_API_KEY="your_alpha_vantage_key"
```

The `.env` file is intentionally ignored by Git.

## Quick Start

Run the native desktop app:

```bash
python swing_trading_system.py
```

The app opens a fixed-sidebar trading terminal. Use the market badge in the sidebar footer or top bar to switch between Korean and US market modes.

## Common Workflows

### Test the KRX API Key

Before requesting Korean market data, verify that the service key works:

```bash
python key_tester.py
```

You can also pass a key just for one run:

```bash
python key_tester.py --key your_data_go_kr_service_key
```

### Download Historical Sample Data

The downloader fetches a predefined list of symbols from Yahoo Finance and writes CSV files to `historical_data/`:

```bash
python download_sp500_data.py
```

The script currently downloads data from `1990-01-01` through `2025-07-27`.

### Train a Model

Start the CLI and choose option `1`:

```bash
python swing_trading_system.py
```

Training reads CSV files from `historical_data/`, creates technical indicators, labels swing opportunities, trains the ensemble, and writes:

- `swing_model_enhanced.pkl`
- `swing_scaler_enhanced.pkl`
- `feature_columns_enhanced.pkl`
- `training_stats.pkl`

### Analyze One Symbol

Start the CLI, choose option `2`, and enter a symbol.

Examples:

- Korean mode: `005930`
- US mode: `AAPL`

The analysis loads the saved model artifacts, fetches recent market data, computes the same feature set used during training, and prints a swing probability, confidence level, current price, stop-loss estimate, and take-profit estimate when enough data is available.

### Monitor Multiple Symbols

Choose option `3` and enter comma-separated symbols.

Examples:

```text
005930,000660,035420
AAPL,MSFT,GOOGL
```

The monitor repeats analysis on an interval and reports symbols whose swing probability crosses the configured alert threshold.

### Screen Symbols from `stocks.json`

Edit `stocks.json` to contain the symbols you want to scan:

```json
[
  "nvda",
  "goog",
  "aapl",
  "msft"
]
```

Open the app and choose `Screener`. The screener analyzes each symbol and shows a ranked summary by swing probability.

### Backtest a Strategy

Open `Backtest`, enter a symbol, and select a lookback window. The backtest walks through historical feature rows, applies the saved model or local fallback scoring, and reports strategy-style performance metrics for the selected period.

## Desktop UI

The dashboard is a native Qt desktop application launched by `python swing_trading_system.py`. It is not a browser wrapper. The sidebar and every screen are created once and retained in a `QStackedWidget`; navigation changes the active page and refreshes its existing labels, tables, and plots in place.

PyQtGraph renders the analysis price chart and backtest equity curves. Provider requests, analysis, training, and backtests run through Qt's thread pool so the interface remains responsive.

Settings are persisted to `trading_ui_config.json`, while API keys are written to `.env` through the Settings screen.

## Building a Standalone Python Executable

Windows:

```powershell
pyinstaller SwingTradingSystem.spec
```

Or use the full command in `pyinstaller_cmd.txt` if you need to tune hidden imports and bundled model artifacts manually.

After building, the executable is written under `dist/`.

## Data and Model Notes

Training expects CSV files in `historical_data/`. The main loader infers symbols from filenames such as:

```text
AAPL_historical.csv
005930_historical.csv
```

The model artifacts are coupled:

- `swing_model_enhanced.pkl` contains the trained estimator.
- `swing_scaler_enhanced.pkl` contains the fitted scaler.
- `feature_columns_enhanced.pkl` defines the expected feature order.
- `training_stats.pkl` stores thresholds, scores, and training metadata.

Keep these files in sync. If you retrain the model, commit or distribute all four artifacts together.

## Security Notes

- Do not commit `.env` or real API keys.
- The Python service layer validates symbols, stock mode values, thresholds, intervals, and data paths before running public operations.
- Public analysis, monitoring, and backtest operations include rate-limit checks.
- The desktop settings screen writes API keys to `.env`, which is ignored by Git.

## Troubleshooting

### `KRX_SERVICE_KEY was not found`

Set `KRX_SERVICE_KEY` in `.env` or as an environment variable, then rerun `python key_tester.py`.

### KRX key fails every endpoint

The key may be incorrect, not approved for the required data.go.kr datasets, or approval may not have propagated yet.

### US analysis fails

Set `ALPHA_VANTAGE_API_KEY` before switching to US mode.

### Model loading fails

Confirm these files exist in the project root:

- `swing_model_enhanced.pkl`
- `swing_scaler_enhanced.pkl`
- `feature_columns_enhanced.pkl`
- `training_stats.pkl`

If they are missing or out of sync, retrain the model from the `Train model` screen.

## Development Notes

- `swing_trading_system.py` contains most backend logic, including data validation, feature engineering, model training, detection, monitoring, screening, and backtesting.
- `download_sp500_data.py` is a separate data-collection utility.
- `key_tester.py` is a focused diagnostic tool for data.go.kr credentials.
- The native desktop UI lives in `swing_trading_system.py` alongside the existing service layer.
- The current repository does not include an automated test suite.

## Disclaimer

This project is experimental trading-research software. Markets are noisy, APIs can fail, and machine-learning predictions can be wrong. Always validate results independently, understand the model assumptions, and never risk money you cannot afford to lose.
