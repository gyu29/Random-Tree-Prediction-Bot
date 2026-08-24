# Random Tree Prediction Bot

A Python swing-trading research system that trains eight regime-specific ensemble classifiers (one per macro-factor category) on historical market data, detects possible swing opportunities, and provides a modern PySide6 desktop terminal for monitoring model settings, watchlists, alerts, analysis, training, and walk-forward backtesting.

> This project is for education and research only. It is not financial advice, and it should not be used as the sole basis for real trading decisions.

## Features

- Trains one hybrid Random Forest + XGBoost classifier per factor category (market beta, growth/tech, small-cap, international/emerging, credit conditions, rates/recession, inflation/safe-haven, energy/commodity) instead of a single one-size-fits-all model.
- Chronological, per-symbol train/validation/test split (`train/<category>/`, `validation/<category>/`, `test/<category>/`) so backtests are genuinely out-of-sample, not a re-run over training data -- and so a model's decision threshold can be tuned on validation without ever touching the data used to report final performance.
- Builds technical-analysis features with `ta`, `pandas`, `numpy`, and scikit-learn; label creation is vectorized (see `tests/test_labeling.py` for the equivalence check against the original loop implementation).
- Model artifacts are versioned and integrity-checked: each `models/<category>/` directory has a `manifest.json` (library versions, hyperparameters, a hash of every training file used, and a code version) plus an HMAC-signed `manifest.sig`, verified before any `.pkl` is unpickled.
- A category's decision threshold can be updated (e.g. after reviewing `scripts/select_thresholds.py`'s output) without retraining -- the trees don't depend on it, only how a predicted probability becomes a trade decision does.
- Supports Korean market lookups through data.go.kr / KRX endpoints, and optional US market lookups through Alpha Vantage, behind a shared provider interface (`app/market_data/`).
- Includes a native PySide6 desktop UI for training, analysis, monitoring, screening, settings, and backtesting.
- Uses cached `QStackedWidget` pages so navigation updates existing widgets instead of rebuilding screens.
- Uses PyQtGraph for responsive price and equity-curve visualization.
- Uses local historical CSV fallbacks so the UI remains useful when provider API keys are unavailable.
- Each watchlist symbol can have its own category override (Settings screen) instead of always relying on automatic symbol-to-category detection.

## Project Layout

```text
Random-Tree-Prediction-Bot/
|-- app/                             # Backend package
|   |-- config.py                    # Paths, constants, .env handling
|   |-- security.py                  # Validation, rate limiting, secrets
|   |-- xml_safety.py                # defusedxml wrapper for KRX responses
|   |-- market_data/                 # KRX + Alpha Vantage providers (shared base)
|   |-- data_loader.py               # CSV loading, category/train/validation/test helpers
|   |-- indicators.py                # Technical-analysis feature engineering
|   |-- labeling.py                  # Vectorized swing-label creation
|   |-- ensemble.py                  # Hybrid RF+XGBoost model
|   |-- trainer.py                   # SwingTradeTrainer
|   |-- model_registry.py            # Signed, versioned model artifacts
|   |-- detector.py                  # Inference + shared walk-forward backtest
|   `-- trading_system.py            # Orchestration facade
|-- ui/                              # PySide6 desktop UI package
|   |-- state.py                     # AppState (thread-safe shared state)
|   |-- widgets.py, tasks.py, app_window.py
|   `-- pages/                       # One file per screen
|-- scripts/
|   |-- build_factor_datasets.py     # Downloads + splits train/validation/test data
|   |-- train_all_categories.py      # Trains + evaluates all 8 categories
|   `-- select_thresholds.py         # Picks decision_threshold per category on validation/
|-- tests/
|   |-- test_labeling.py             # Vectorized-vs-loop labeling equivalence
|   |-- test_model_registry.py       # save/load signature + tamper/corruption rejection
|   |-- test_detector.py             # walk_forward_backtest entry/exit/equity logic
|   `-- test_security.py             # SecurityValidator + RequestRateLimiter
|-- train/<category>/<symbol>.csv        # Oldest 55% per symbol (generated) -- the only
|                                         # data any model is ever fit on
|-- validation/<category>/<symbol>.csv   # Next 15% per symbol (generated) -- for choosing
|                                         # things like decision_threshold without
|                                         # touching test/
|-- test/<category>/<symbol>.csv         # Most recent 30% per symbol (generated) -- touched
|                                         # exactly once, to report final performance
|-- models/<category>/               # model.pkl, scaler.pkl, features.pkl,
|                                     # training_stats.pkl, manifest.json,
|                                     # manifest.sig (generated)
|-- docs/                            # Promotional landing page (static, no build step)
|   |-- index.html
|   |-- styles.css
|   |-- script.js
|   `-- fonts/*.woff2                # Vendored locally, not loaded from a font CDN
|-- main.py                          # Entry point
|-- qt_trading_ui.py                 # Thin shim -> ui.app_window
|-- key_tester.py                    # Tests data.go.kr KRX service-key access
`-- .env.example                     # Example runtime secrets file
```

`train/`, `validation/`, `test/`, and `models/` are gitignored: they're regenerated by `scripts/build_factor_datasets.py` and `scripts/train_all_categories.py` rather than committed.

## Requirements

### Python

- Python 3.10 or newer recommended
- `pip`

Install the Python packages used by the scripts:

```bash
pip install -r requirements.txt
```

`requirements.txt` pins exact versions (see the comment at its top for why -- it mirrors the same reproducibility reasoning as model manifests). `xgboost` is required for training. `yfinance` is only needed when running `scripts/build_factor_datasets.py`. `defusedxml` is used to parse KRX API responses safely. `pytest` is only needed to run `tests/`. `pyinstaller` is only needed when building an executable.

## Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

Then edit `.env`:

```env
KRX_SERVICE_KEY=your_data_go_kr_service_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
```

`KRX_SERVICE_KEY` is used for Korean market data. `ALPHA_VANTAGE_API_KEY` is optional and only needed for live US market lookups. A third value, `MODEL_SIGNING_KEY`, is generated automatically the first time a model is trained and saved to the same file -- it signs model manifests (see [Model Reproducibility and Integrity](#model-reproducibility-and-integrity)) and should never be shared or committed.

The `.env` file is intentionally ignored by Git.

## Quick Start

```bash
python main.py
```

The app opens a fixed-sidebar trading terminal. Use the market badge in the sidebar footer or top bar to switch between Korean and US market modes.

## Common Workflows

### Test the KRX API Key

```bash
python key_tester.py
# or, for a one-off key:
python key_tester.py --key your_data_go_kr_service_key
```

### Build the Factor Datasets

Downloads all 56 tickers across the 8 factor categories from Yahoo Finance and writes the chronological per-symbol split: oldest 55% to `train/`, next 15% to `validation/`, most recent 30% to `test/`.

```bash
python scripts/build_factor_datasets.py
```

### Train and Evaluate All 8 Category Models

```bash
python scripts/train_all_categories.py
```

For each category this trains on `train/<category>/*.csv`, saves a signed model to `models/<category>/`, and evaluates it on the held-out `test/<category>/*.csv` split with a walk-forward backtest, printing a per-category and combined summary.

You can also train a single category from the desktop app's `Train model` screen, which exposes the same hyperparameters (RF estimators, XGBoost learning rate/depth, swing window, swing threshold) per category. `decision_threshold` isn't one of them -- every category trains with the same default (see below for why, and how to change it after the fact).

### Select a Decision Threshold from Validation

`decision_threshold` is the minimum predicted probability required to actually enter a trade. Choosing it by testing candidate values against `test/` and keeping whichever looks best is a form of overfitting -- it's guaranteed to look good on the data it was chosen from, whether or not it generalizes.

```bash
python scripts/select_thresholds.py
```

For each category, this sweeps candidate thresholds against `validation/` only, picks the one with the best pooled Sharpe (requiring at least 5 trades to trust the pick over the existing default), then reports that threshold's performance on both validation and `test/` side by side -- the honest check of whether it actually holds up on data the selection process never saw. In practice this is a mixed bag: it improved risk-adjusted performance for some categories and made others worse, including one case where a validation-selected threshold that looked reasonable on validation lost money on test. Read the printed comparison before trusting any single category's number.

See [MODEL_CALIBRATION_FINDINGS.md](MODEL_CALIBRATION_FINDINGS.md) for a worked example of this failure mode across all 8 categories -- why four of them were trading zero times at all, a stricter validation/test-agreement search than this script's own selection rule, and the category-appropriate `swing_threshold` retrain that ended up fixing most of them.

To apply a threshold you've decided to keep, without retraining (the RF/XGBoost trees don't depend on it):

```python
from app import model_registry
model_registry.update_decision_threshold("growth_tech", 0.70)
```

### Analyze One Symbol

Open `Analyze`, enter a symbol, and pick a market and factor category (auto-suggested from the symbol when recognized). The analysis loads that category's model, fetches recent market data, computes the same feature set used during training, and shows a swing probability, confidence level, current price, stop-loss, and take-profit. The probability metric is colored by the actual prediction direction, not raw confidence -- teal only for a positive swing call, red for a confident *negative* call, muted gray otherwise, since "high confidence" alone doesn't say which way the model is confident. If nothing has been trained yet for that category, a clearly-labeled heuristic estimate is shown instead -- it is never presented as if it came from the model.

### Monitor and Screen

`Monitor` and `Screener` operate over your watchlist (`Settings` screen); each symbol is automatically scored with its own factor category's model. Add symbols and, optionally, override which category's model scores each one -- pick `(auto-detect)` to keep tracking the automatic symbol-to-category mapping instead of freezing in a choice, or pick one of the 8 categories explicitly.

### Backtest a Strategy

Open `Backtest`. Choose any CSV (not just the curated 56 -- any file with OHLCV columns works) and, independently, a Model to score it with; picking a file auto-suggests a Model as a convenience, but the two are fully decoupled, so you can score any file against any category's model. The threshold dropdown automatically shows the selected model's actual trained default whenever you change categories (falling back to the global default if that category isn't trained), and you can override it before running. Choose a lookback window and run the walk-forward backtest.

## Desktop UI

The dashboard is a native Qt desktop application launched by `python main.py`. It is not a browser wrapper. The sidebar and every screen are created once and retained in a `QStackedWidget`; navigation changes the active page and refreshes its existing labels, tables, and plots in place.

PyQtGraph renders the analysis price chart and backtest equity curves. Provider requests, analysis, training, and backtests run through Qt's thread pool so the interface remains responsive; shared state (the active model per category, the watchlist, cached market data) is guarded by a single lock in `ui/state.py` so concurrent background tasks can't race on it.

Settings are persisted to `trading_ui_config.json`, while API keys are written to `.env` through the Settings screen.

## Building a Standalone Python Executable

There is no maintained PyInstaller spec for the current layout. An earlier `pyinstaller_cmd.txt` targeted the pre-rewrite flat `swing_trading_system.py` entry point and single-model `.pkl` files and was removed as stale. A working command for the current app would need to target `main.py`, add hidden imports for `app.*`/`ui.*`, and bundle `models/<category>/` rather than loose `.pkl` files at the root.

## Data and Model Notes

Training expects CSV files in `train/<category>/`, one per symbol, named `<SYMBOL>.csv` (e.g. `train/growth_tech/AAPL.csv`). `validation/<category>/` and `test/<category>/` mirror the same layout. `scripts/build_factor_datasets.py` builds all three for you, split chronologically per symbol so train ends before validation begins, and validation ends before test begins -- never a random shuffle.

Each `models/<category>/` directory holds:

- `model.pkl` -- the trained hybrid ensemble.
- `scaler.pkl` -- the fitted feature scaler.
- `features.pkl` -- the expected feature order.
- `training_stats.pkl` -- thresholds, scores, and training metadata.
- `manifest.json` / `manifest.sig` -- see below.

These five files are coupled and always rewritten together atomically, whether by a full retrain or by `model_registry.update_decision_threshold()`.

## Model Reproducibility and Integrity

Every `models/<category>/manifest.json` records what produced that model: library versions (Python/pandas/numpy/scikit-learn/xgboost/joblib), the full hyperparameters used, a per-file sha256 + row count + date range for every training CSV, and a code version (the git commit hash, when available). `manifest.sig` is an HMAC-SHA256 over that manifest, keyed by the local `MODEL_SIGNING_KEY` in `.env`.

Before loading any `.pkl`, `app/model_registry.py` recomputes and checks both the manifest signature and each artifact's hash, refusing to load on any mismatch. This is tamper-evidence for a local, single-user tool -- it catches accidental corruption and casual tampering by anyone without your local signing key -- not a substitute for a real code-signing PKI.

`model_registry.update_decision_threshold(category, new_threshold)` is the one supported way to modify a trained model after the fact: it re-saves through the same signed path (so the manifest/signature/hashes stay consistent) rather than leaving the artifacts in a state `load()` would correctly refuse as tampered. Everything else about the model -- the trees, the scaler, the feature list -- is unchanged; only `decision_threshold` (in `training_stats`, the model object, and the manifest's hyperparameters record) is updated.

## Security Notes

- Do not commit `.env` or real API keys (or `MODEL_SIGNING_KEY`).
- `app/security.py` validates symbols, stock mode values, thresholds, intervals, and data paths before running public operations, and applies rate limits per operation.
- KRX API responses are parsed with `defusedxml` rather than the standard library's `xml.etree.ElementTree`, which is vulnerable to entity-expansion ("billion laughs") payloads.
- Model artifacts are signature- and hash-checked before unpickling (see above).
- The desktop settings screen writes API keys to `.env`, which is ignored by Git.

## Troubleshooting

### `KRX_SERVICE_KEY was not found`

Set `KRX_SERVICE_KEY` in `.env` or as an environment variable, then rerun `python key_tester.py`.

### KRX key fails every endpoint

The key may be incorrect, not approved for the required data.go.kr datasets, or approval may not have propagated yet.

### US analysis fails

Set `ALPHA_VANTAGE_API_KEY` before switching to US mode.

### Model loading fails / "No trained model available for category"

Confirm `models/<category>/` exists and contains all five files listed above. If a manifest signature or artifact hash check fails, the category needs to be retrained (someone/something modified the files after they were saved). Retrain from the `Train model` screen or `python scripts/train_all_categories.py`.

## Development Notes

- `app/` contains the backend: security/validation, market data providers, feature engineering, labeling, training, the signed model registry, inference, and orchestration. See the module docstrings for the reasoning behind the less-obvious design choices (the label threshold, the class-imbalance handling, the walk-forward split).
- `ui/` contains the PySide6 desktop UI, split into state/widgets/tasks/pages.
- `qt_trading_ui.py` is now a thin compatibility shim over `ui.app_window`; `main.py` is the primary entry point.
- `key_tester.py` remains an intentionally standalone diagnostic tool for data.go.kr credentials (no dependency on `app/`, so it still works if the main app doesn't import).
- `tests/test_labeling.py` verifies the vectorized label implementation against the original row-by-row loop it replaced. `tests/test_model_registry.py` covers save/load's signature and per-artifact hash verification, including that tampering, corruption, or a changed `MODEL_SIGNING_KEY` are rejected rather than silently loaded. `tests/test_detector.py` covers `walk_forward_backtest`'s entry/exit/equity simulation (stop-loss, take-profit, max-time exits, and the decision-window boundary) using a fake model/scaler so trade-triggering bars are deterministic instead of depending on what a real trained ensemble predicts. `tests/test_security.py` covers `SecurityValidator` (symbol/US and KR regex boundaries, numeric validation's bool- and NaN-rejection, request-context handling) and `RequestRateLimiter`'s sliding-window bucketing, using a fake clock so window-boundary timing is exact rather than racing the real clock. Run with `pytest tests/`.
- `app/detector.py`'s `walk_forward_backtest` batches its probability predictions once per call rather than per bar; this matters more than it sounds like it should, since a naive per-bar implementation is slow enough on a large symbol's history to make routine re-evaluation (`train_all_categories.py`, `select_thresholds.py`) impractical.
- `docs/` is an unrelated promotional page: `index.html` + `styles.css` + `script.js` + `fonts/*.woff2`, no build step, no third-party requests (fonts are vendored locally, not loaded from a font CDN). Open `docs/index.html` directly in a browser, or point GitHub Pages at the `docs/` folder to host it. It isn't part of the application and imports nothing from `app/`/`ui/`.

## Disclaimer

This project is experimental trading-research software. Markets are noisy, APIs can fail, and machine-learning predictions can be wrong. Always validate results independently, understand the model assumptions, and never risk money you cannot afford to lose.
