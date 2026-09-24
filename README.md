# Random Tree Prediction Bot

A Python swing-trading research system that trains eight regime-specific ensemble classifiers (one per macro-factor category) on historical market data, detects possible multi-month swing opportunities, and provides a PySide6 desktop terminal for monitoring model settings, watchlists, alerts, analysis, training, and walk-forward backtesting.

Positions are held for **three to six months**: a trade exits between 63 and 126 trading sessions after entry, and a label asks whether the median close over that window was up by the category's threshold. The project moved to this horizon from 3-10 days in September 2026; a six-to-twelve month hold was measured first and rejected (see `app/config.py`).

**None of these eight models currently produces a signal this project can stand behind, and the app says so.** Each category is checked against the alternative of ignoring it entirely: bin every trade the model would open with no threshold at all by its predicted probability, and see whether higher-probability trades reliably earn more (`scripts/expected_value_thresholds.py`). The significance of that difference is measured by resampling blocks of consecutive entry dates, because trades are not independent -- several fire the same day across correlated symbols, and each is held long enough to overlap the ones after it.

On the frozen dataset (`paper/results/tables.md`), two of the eight categories produce a probability floor on validation, and neither beats ignoring the model out of sample: `international_emerging` by +3.14% a trade at +1.71 standard errors, `small_cap` by +3.68% at +1.34, against a bar of 2. The other four calibrated categories take too few trades on validation (105-181) to estimate the curve at all, and two cannot be calibrated. All eight are listed in `app.config.CATEGORIES_FAILING_VALIDATION`. They still load and can still be analyzed, because investigating a model means being able to run it, but they raise no alerts, take no place in the screener ranking, and carry a warning on every result.

That is a statement about evidence, not about the models: eight overlapping factor categories do not yield enough independent observations to separate a real edge from noise at a multi-month hold. Effective independent series per category run 1.04 to 3.70 across 162 tickers, and the whole history of a category holds roughly 53 to 137 independent holding-period outcomes. Adding more correlated tickers does not help. Treat all of it as research, not advice.

> This project is for education and research only. It is not financial advice, and it should not be used as the sole basis for real trading decisions.

## Features

- Trains one hybrid Random Forest + XGBoost classifier per factor category (market beta, growth/tech, small-cap, international/emerging, credit conditions, rates/recession, inflation/safe-haven, energy/commodity) instead of a single one-size-fits-all model.
- Each member's probabilities are calibrated on a held-out slice of the training data before the two are averaged, so predicted probabilities are comparable and can order trades, and the entry threshold is computed from realized returns instead of swept for (`app/ensemble.py`, `scripts/expected_value_thresholds.py`). Calibration does not survive out of sample: on the test split the models under-predict badly -- `small_cap`'s top bin, predicted at about 1.5%, came true about 24% of the time (`paper/results/figures/fig6_test_reliability.png`). Read the probabilities as a ranking, not as odds.
- Features include market context — VIX, the term spread, and each symbol's return and beta relative to a benchmark (`app/market_context.py`) — so a model can distinguish "this name became volatile" from "everything became volatile".
- 162 tickers across the eight categories, chosen as distinct exposures (sectors, single countries, credit tiers, curve maturities) rather than near-duplicate wrappers, and filtered to funds liquid enough to actually trade. `market_beta` once held seven funds tracking the same index at a 0.98 median pairwise return correlation. Anything turning over under $5M a day was removed: a position in a fund trading $41,000 a day cannot be taken at size, so returns measured on it are not returns anybody could have had. The universe is ETFs except `growth_tech`, which carries individual mega-cap stocks and is survivorship-biased by construction.
- Chronological, calendar-aligned train/validation/test split (`train/<category>/`, `validation/<category>/`, `test/<category>/`): one pair of cutoff dates per category, applied to every symbol in it, so backtests are genuinely out-of-sample -- and so a model's decision threshold can be chosen on validation without ever touching the data used to report final performance. The cutoff is per category rather than per symbol because a category's symbols move together; splitting each symbol's own rows by percentage puts up to 52% of a category's test rows on dates the model already trained on through a sibling ticker (`paper/results/tables.md`, T3). Both seams carry a 126-row embargo, since a label reads 126 sessions into the future.
- Purged walk-forward cross-validation (`scripts/walk_forward_cv.py`) reports per-category ranking quality across five expanding-window folds rather than a single number.
- A reproducible paper run (`scripts/paper_run.py`) regenerates every reported table and figure from a hash-verified data snapshot. See [Paper and Reproducible Results](#paper-and-reproducible-results).
- Builds technical-analysis features with `ta`, `pandas`, `numpy`, and scikit-learn; label creation is vectorized (see `tests/test_labeling.py` for the equivalence check against the original loop implementation).
- Model artifacts are versioned and integrity-checked: each `models/<category>/` directory has a `manifest.json` (library versions, hyperparameters, a hash of every training file used, and a code version) plus an HMAC-signed `manifest.sig`, verified before any `.pkl` is unpickled.
- A category's decision threshold can be updated without retraining -- the trees don't depend on it, only how a predicted probability becomes a trade decision does.
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
|   |-- config.py                    # Paths, horizon, universe, thresholds, validation gate
|   |-- security.py                  # Validation, rate limiting, secrets
|   |-- xml_safety.py                # defusedxml wrapper for KRX responses
|   |-- market_data/                 # KRX + Alpha Vantage providers (shared base)
|   |-- market_context.py            # VIX / benchmark / rates context features
|   |-- data_loader.py               # CSV loading, category/train/validation/test helpers
|   |-- indicators.py                # Technical-analysis feature engineering
|   |-- labeling.py                  # Vectorized swing-label creation
|   |-- ensemble.py                  # Hybrid RF+XGBoost model with calibration
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
|   |-- train_all_categories.py      # Trains all 8 categories, tests each against its null
|   |-- expected_value_thresholds.py # Derives each decision floor on validation/
|   |-- walk_forward_cv.py           # Purged expanding-window cross-validation
|   |-- select_thresholds.py         # Earlier threshold sweep; shared loading helpers
|   `-- paper_run.py                 # Regenerates every table and figure in paper/
|-- paper/
|   |-- README.md                    # How to reproduce, what each output is
|   |-- data_manifest.json           # SHA-256 of every data file in the frozen snapshot
|   `-- results/                     # Tables, figures, trades, run manifest (generated)
|-- tests/                           # pytest suite (see Development Notes)
|-- train/<category>/<symbol>.csv        # Before the category's train cutoff (generated)
|                                         # -- the only data any model is ever fit on
|-- validation/<category>/<symbol>.csv   # Between the two cutoffs (generated) -- for
|                                         # choosing the decision floor without
|                                         # touching test/
|-- test/<category>/<symbol>.csv         # After the category's validation cutoff
|                                         # (generated) -- touched once, to report
|                                         # final performance
|-- models/<category>/               # model.pkl, scaler.pkl, features.pkl,
|                                     # training_stats.pkl, manifest.json,
|                                     # manifest.sig (generated)
|-- docs/                            # Promotional landing page (static, no build step)
|   |-- 2026-08-24-calibration-*.md  # Archived investigation, superseded (see below)
|   |-- index.html
|   |-- styles.css
|   |-- script.js
|   `-- fonts/*.woff2                # Vendored locally, not loaded from a font CDN
|-- main.py                          # Entry point
|-- qt_trading_ui.py                 # Thin shim -> ui.app_window
|-- key_tester.py                    # Tests data.go.kr KRX service-key access
`-- .env.example                     # Example runtime secrets file
```

`train/`, `validation/`, `test/`, `market_context.csv`, and `models/` are gitignored: they're regenerated by `scripts/build_factor_datasets.py` and `scripts/train_all_categories.py` rather than committed. `paper/results/` is committed, because it is the record of what the paper reports.

## Requirements

### Python

- Python 3.10 or 3.11. The pinned `pandas==2.0.3` has no wheels for 3.12 or later. The saved models and the paper run were produced on 3.11.3.
- `pip`

Install the Python packages used by the scripts:

```bash
pip install -r requirements.txt
```

`requirements.txt` pins exact versions (see the comment at its top for why -- it mirrors the same reproducibility reasoning as model manifests). `xgboost` is required for training. `yfinance` is only needed when running `scripts/build_factor_datasets.py`. `defusedxml` is used to parse KRX API responses safely. `matplotlib` and `scipy` are only needed for `scripts/paper_run.py`. `pytest` is only needed to run `tests/`. `pyinstaller` is only needed when building an executable.

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

Downloads all 162 tickers across the 8 factor categories from Yahoo Finance and writes the calendar-aligned split. Validation and test are each sized by what they have to hold at this horizon -- a 200-session feature warm-up, room for the last entry to reach its exit, and six non-overlapping holding periods for the block bootstrap to resample -- and train receives everything before them. The same two cutoff dates apply to every symbol in a category, with 126 rows dropped at each seam as an embargo. On the current snapshot every category splits identically: train to 2017-01-24, validation 2017-07-26 to 2021-11-09, test 2022-05-12 to 2026-09-03.

Because the cutoffs are derived from whatever history Yahoo returns that day, they move between runs, and Yahoo also revises adjusted prices after every dividend. Pin the cutoffs in `app.config.CALENDAR_SPLIT_CUTOFFS` to keep the split stable:

```python
CALENDAR_SPLIT_CUTOFFS = {"growth_tech": ("2017-07-26", "2022-05-12")}
```

Pinning the cutoffs does not pin the prices. For a dataset that is reproducible byte for byte, keep the files and their hashes -- see [Paper and Reproducible Results](#paper-and-reproducible-results).

A symbol whose history begins after its category's train cutoff contributes no training rows. The run reports that rather than writing a near-empty file, and training excludes any symbol too short to compute the full feature set.

```bash
python scripts/build_factor_datasets.py
```

### Train and Evaluate All 8 Category Models

```bash
python scripts/train_all_categories.py
```

For each category this trains on `train/<category>/*.csv` with the category's swing threshold from `app.config.CALIBRATED_SWING_THRESHOLDS`, saves a signed model to `models/<category>/`, applies the category's decision floor from `CALIBRATED_DECISION_THRESHOLDS` if it has one, and evaluates it on `test/<category>/*.csv`. The test evaluation reports PR-AUC and ROC-AUC, then backtests the model at its floor and again with the threshold at zero -- the same entries and exits with the model's ranking ignored -- and reports the difference in block-bootstrap standard errors.

You can also train a single category from the desktop app's `Train model` screen, which exposes the same hyperparameters (RF estimators, XGBoost learning rate/depth, minimum hold, swing window, swing threshold) per category. It trains with the default decision threshold; apply a derived floor afterwards as below.

### Derive a Decision Floor from Validation

`decision_threshold` is the minimum predicted probability required to enter a trade. Choosing it by trying values against `test/` and keeping whichever looks best is overfitting -- it is guaranteed to look good on the data it was chosen from.

```bash
python scripts/expected_value_thresholds.py
```

For each calibrated category this takes every trade the model would open on `validation/` with no threshold, bins the trades by predicted probability, and asks whether trades above a candidate floor out-earn those below it. A floor is kept only if the best candidate's separation, measured in block-bootstrap standard errors, beats the 95th percentile of the best-of-candidates statistic under a block permutation -- a bar that pays for having searched. Where no floor clears it, or a category takes too few trades to estimate the curve, the script says so rather than printing a number. It prints any floor it finds at full precision; copy it exactly into `CALIBRATED_DECISION_THRESHOLDS`, because calibrated probabilities sit on plateaus and a rounded floor can land on the other side of one. `small_cap`'s was once copied as 0.001289 instead of 1/776, which excluded 35 of the 119 validation trades it was chosen on and reported +2.33 standard errors for a rule that scores +1.34.

`scripts/select_thresholds.py` is the earlier approach, a sweep over a fixed grid scored on a shrunk mean return. It still runs, and its loading helpers are shared, but the floors in `app/config.py` come from `expected_value_thresholds.py`.

See [the archived 2026-08-24 calibration investigation](docs/2026-08-24-calibration-investigation.md) for the earlier work on swing thresholds. Its numbers predate the current horizon, split, universe and feature set and describe nothing on disk; its reasoning still holds.

To apply a floor without retraining (the RF/XGBoost trees don't depend on it):

```python
from app import model_registry
from app.config import CALIBRATED_DECISION_THRESHOLDS
model_registry.update_decision_threshold("small_cap", CALIBRATED_DECISION_THRESHOLDS["small_cap"])
```

### Cross-Validate

```bash
python scripts/walk_forward_cv.py
```

Five expanding-window folds per category, cut from `train/` and `validation/` only so `test/` stays untouched, with the same embargo at every fold boundary. Reports PR-AUC and ROC-AUC per fold, so a category that ranks well in one regime and badly in the next shows up as such.

### Analyze One Symbol

Open `Analyze`, enter a symbol, and pick a market and factor category (auto-suggested from the symbol when recognized). The analysis loads that category's model, fetches recent market data, computes the same feature set used during training, and shows a swing probability, confidence level, current price, stop-loss, and take-profit. The probability metric is colored by the actual prediction direction, not raw confidence -- teal only for a positive swing call, red for a confident *negative* call, muted gray otherwise, since "high confidence" alone doesn't say which way the model is confident. Every result from a category in `CATEGORIES_FAILING_VALIDATION` carries its warning. If nothing has been trained yet for that category, a clearly-labeled heuristic estimate is shown instead -- it is never presented as if it came from the model.

### Monitor and Screen

`Monitor` and `Screener` operate over your watchlist (`Settings` screen); each symbol is automatically scored with its own factor category's model. Add symbols and, optionally, override which category's model scores each one -- pick `(auto-detect)` to keep tracking the automatic symbol-to-category mapping instead of freezing in a choice, or pick one of the 8 categories explicitly. Categories that failed validation raise no alerts and are left out of the screener ranking.

### Backtest a Strategy

Open `Backtest`. Choose any CSV (not just the curated 162 -- any file with OHLCV columns works) and, independently, a Model to score it with; picking a file auto-suggests a Model as a convenience, but the two are fully decoupled, so you can score any file against any category's model. The threshold dropdown automatically shows the selected model's actual trained default whenever you change categories (falling back to the global default if that category isn't trained), and you can override it before running. Choose a lookback window and run the walk-forward backtest. At a 126-session hold, a window needs well over a year of data to produce any completed trades.

## Paper and Reproducible Results

`paper/` holds everything behind the write-up of these results. `scripts/paper_run.py` regenerates every table and figure from one frozen dataset:

```bash
python scripts/paper_run.py --require-clean
```

- The dataset is the 2026-09-04 snapshot. `paper/data_manifest.json` records a SHA-256 for each of its 484 files, and the run refuses to start if any differs.
- Models are refit in memory with the project's own trainer and seeds and never written to `models/`. The refit matches the saved models to within 1e-14.
- Floors are derived on validation by `expected_value_thresholds.solve_threshold`, and test is scored once against the model-off null.
- `--require-clean` refuses to run with uncommitted code changes, so `paper/results/run_manifest.json` names the exact commit that produced the numbers.

The run takes about 17 minutes. `paper/README.md` lists every table and figure and the question each one answers. The price data itself is not in the repository: keep an archived copy that matches the manifest, because a fresh download will not.

## Desktop UI

The dashboard is a native Qt desktop application launched by `python main.py`. It is not a browser wrapper. The sidebar and every screen are created once and retained in a `QStackedWidget`; navigation changes the active page and refreshes its existing labels, tables, and plots in place.

PyQtGraph renders the analysis price chart and backtest equity curves. Provider requests, analysis, training, and backtests run through Qt's thread pool so the interface remains responsive; shared state (the active model per category, the watchlist, cached market data) is guarded by a single lock in `ui/state.py` so concurrent background tasks can't race on it.

Settings are persisted to `trading_ui_config.json`, while API keys are written to `.env` through the Settings screen.

## Building a Standalone Python Executable

There is no maintained PyInstaller spec for the current layout. An earlier `pyinstaller_cmd.txt` targeted the pre-rewrite flat `swing_trading_system.py` entry point and single-model `.pkl` files and was removed as stale. A working command for the current app would need to target `main.py`, add hidden imports for `app.*`/`ui.*`, and bundle `models/<category>/` rather than loose `.pkl` files at the root.

## Data and Model Notes

Training expects CSV files in `train/<category>/`, one per symbol, named `<SYMBOL>.csv` (e.g. `train/growth_tech/AAPL.csv`). `validation/<category>/` and `test/<category>/` mirror the same layout. `scripts/build_factor_datasets.py` builds all three for you, split at two calendar cutoffs shared by every symbol in the category, so train ends before validation begins and validation ends before test begins -- for the category as a whole, not just per symbol, and never a random shuffle.

Labels are `terminal`: a row is positive when the median close over sessions 63 to 126 after it is up by the category's swing threshold (`app.config.CALIBRATED_SWING_THRESHOLDS`, 10% for credit up to 50% for growth_tech). The alternative, `peak`, asks whether the high ever touched the threshold inside the window; at this horizon it labels 2.3 to 3.1 times as many rows positive, most of them excursions a holder never sold into.

Model features are all scale-free -- ratios, widths, positions, slopes, percentages of price, and rolling z-scores. Absolute price and volume levels (`sma_*`, `ema_*`, the Bollinger band edges, `atr_14`, `macd`, `on_balance_volume`, and the rest of `app.indicators.PRICE_LEVEL_COLUMNS`) are still computed, because stop-loss sizing and `macd_crossover` need them, but `app/trainer.py` excludes them from the feature set. A tree can only split at values it saw while fitting, and a symbol's price level in the test window routinely sits outside its entire training range: AAPL trains under a dollar and is tested above $100.

Accuracy is not a meaningful score here and the training output doesn't lead with it. Positive labels run roughly 3-15% of rows depending on category and period, so "never predict a swing" scores 85-97%. `train()` and `scripts/train_all_categories.py` report PR-AUC against the base rate, ROC-AUC, precision and recall, and print accuracy only beside the always-negative baseline it has to beat. Even those ranking scores overstate what is known: test rows overlap by up to 99% of their outcome window, so a test-split AUC describes one path through history rather than many independent outcomes.

Each `models/<category>/` directory holds:

- `model.pkl` -- the trained hybrid ensemble.
- `scaler.pkl` -- the fitted feature scaler.
- `features.pkl` -- the expected feature order.
- `training_stats.pkl` -- thresholds, scores, and training metadata.
- `manifest.json` / `manifest.sig` -- see below.

These files are coupled and always rewritten together atomically, whether by a full retrain or by `model_registry.update_decision_threshold()`. The app reads a category's decision threshold from its saved model, not from `app/config.py`, so a changed floor in config takes effect only once it is applied to the model.

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

Confirm `models/<category>/` exists and contains all the files listed above. If a manifest signature or artifact hash check fails, the category needs to be retrained (someone/something modified the files after they were saved). Retrain from the `Train model` screen or `python scripts/train_all_categories.py`.

### `paper_run.py` refuses to start

"data on disk is not the frozen dataset" means a data file differs from `paper/data_manifest.json` -- usually because the datasets were rebuilt from a fresh download. Restore the archived snapshot. "Refusing: the working tree has uncommitted changes" comes from `--require-clean`; commit first, or drop the flag for a run that will be stamped as not citable.

## Development Notes

- `app/` contains the backend: security/validation, market data providers, feature engineering, labeling, training, the signed model registry, inference, and orchestration. See the module docstrings for the reasoning behind the less-obvious design choices (the label, the class-imbalance handling, the walk-forward split, the calibration).
- `ui/` contains the PySide6 desktop UI, split into state/widgets/tasks/pages.
- `qt_trading_ui.py` is a thin compatibility shim over `ui.app_window`; `main.py` is the primary entry point.
- `key_tester.py` remains an intentionally standalone diagnostic tool for data.go.kr credentials (no dependency on `app/`, so it still works if the main app doesn't import).
- `tests/` covers labeling (vectorized against the original loop), the model registry's signature and hash checks, the backtest's entry/exit/equity logic, the security validators and rate limiter, market-context alignment, the calendar split and its horizon-derived sizing, calibration and the floor search, the validation gate, and walk-forward cross-validation. Run with `pytest tests/`.
- `app/detector.py`'s `walk_forward_backtest` batches its probability predictions once per call rather than per bar; this matters more than it sounds like it should, since a naive per-bar implementation is slow enough on a large symbol's history to make routine re-evaluation impractical.
- `docs/` holds an unrelated promotional page -- `index.html` + `styles.css` + `script.js` + `fonts/*.woff2`, no build step, no third-party requests (fonts are vendored locally, not loaded from a font CDN). Open `docs/index.html` directly in a browser, or point GitHub Pages at the `docs/` folder to host it. It isn't part of the application and imports nothing from `app/`/`ui/`. The dated Markdown file alongside it is an archived investigation, kept for its reasoning; it describes an earlier horizon, split, feature set and universe, and none of its numbers match what is on disk today.

## Disclaimer

This project is experimental trading-research software. Markets are noisy, APIs can fail, and machine-learning predictions can be wrong. Always validate results independently, understand the model assumptions, and never risk money you cannot afford to lose.
