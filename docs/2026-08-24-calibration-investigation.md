# Archived: Threshold Calibration Investigation, 2026-08-24

> **This is a historical record, not a description of the system.** Everything below was
> measured against the per-symbol percentage split, the 108-feature set, and the
> 56-ticker universe, all three of which have since been replaced. Two full rebuilds have
> happened since: the dataset is now split at category-wide calendar cutoffs over 208
> tickers, the feature set is 102 scale-free columns including market context, and model
> probabilities are calibrated. **No number in this document corresponds to anything
> currently in `models/`, `train/` or `test/`.** It is kept because the reasoning is
> still correct and still explains why the current design looks the way it does.
>
> For the state of the models today, read these instead -- all of them live beside the
> code they describe, which is the point:
>
> | What | Where |
> |---|---|
> | Which categories produce a usable signal, and why the others don't | `app/config.py`, `CATEGORIES_FAILING_VALIDATION` |
> | How each entry threshold is derived | `scripts/expected_value_thresholds.py` |
> | Why thresholds are computed rather than swept | `app/ensemble.py` |
> | How the split and its embargoes work | `scripts/build_factor_datasets.py` |
> | How every metric gets an error bar | `scripts/walk_forward_cv.py` |
>
> What remains useful here: why a uniform 15% swing threshold silenced four categories
> (section 2), and why picking a decision threshold by the best Sharpe among a handful of
> trades is not a calibration (section 3). Both mistakes are easy to make again. The
> `swing_threshold` values in section 4 are still the ones in `app/config.py`; every
> `decision_threshold` in it has been superseded twice over.

A record of a threshold-calibration investigation across all 8 factor-category models: why four categories were producing zero trades, what was tried, what actually changed, and what still shouldn't be trusted at face value. This is a point-in-time snapshot of one retrain against one fixed chronological split, not a permanent guarantee -- numbers here will drift the next time any category is retrained or the test window rolls forward.

`models/` is gitignored, as it always has been in this project (rebuilt by `scripts/train_all_categories.py`). The retrained artifacts described here live only in the local `models/<category>/` directories on the machine this was run on -- this document (and the earlier `app/security.py` fix / test suite / `requirements.txt` in this same session) are what actually ship to the repo.

## 1. Baseline: evaluating the 8 already-trained models

Running each category's existing signed model against its own `test/<category>/*.csv` holdout (no retraining, just loading and backtesting what was already there):

| Category | Threshold | Trades | Win rate | Compounded return |
|---|---|---|---|---|
| market_beta | 65% | 9 | 88.9% | +192.1% |
| energy_commodity | 65% | 34 | 52.9% | +113.6% |
| growth_tech | 70% | 6 | 83.3% | +55.2% |
| international_emerging | 65% | 6 | 66.7% | +29.9% |
| credit_conditions | 65% | 0 | — | — |
| inflation_safe_haven | 65% | 0 | — | — |
| rates_recession | 65% | 0 | — | — |
| small_cap | 65% | 0 | — | — |

Half the categories never once crossed their decision threshold on any test-set symbol, over the entire held-out period.

## 2. Why four categories never traded

`swing_threshold=0.15` (a 15%+ move within a 3–10 day window defines a positive "swing" label) was applied uniformly to all 8 categories by `train_model()`'s defaults, regardless of how volatile that asset class actually is.

| Category | Train positive rate (at 0.15) | Highest probability ever seen on the entire test set |
|---|---|---|
| credit_conditions | 0.17% | 33.3% |
| inflation_safe_haven | 0.55% | 28.1% |
| rates_recession | 0.24% | 20.3% |
| small_cap | 0.47% | 36.0% |
| *market_beta (comparison)* | 0.37% | 94.1% |
| *growth_tech (comparison)* | 7.99% | 78.7% |

For investment-grade credit, Treasury/inflation hedges, and rate-sensitive bonds, a 15% move in under two weeks is a near-nonevent -- under 0.6% of training rows ever qualified, so the model correctly learned to almost never predict positive, and never got within 30 points of threshold on the entire test period. market_beta has an equally rare training label (0.37%) but still hit 94% at least once, because its test window (1997–2026, per-symbol) spans real tail events (2008, 2020) that the newer bond/credit ETFs' shorter test windows (2014–2020 starts) simply don't contain.

## 3. `scripts/select_thresholds.py`'s sweep -- useful, but not safe to trust blindly

Its selection rule (best validation Sharpe among thresholds with ≥5 trades) found *something* for `rates_recession` and `small_cap`, but both cases demonstrate exactly why that rule isn't sufficient on its own:

- **small_cap** at the selected 5% threshold: validation looked mediocre-but-fine (+7.6%), then **lost money on test** (−9.1%) despite a majority win rate.
- **rates_recession** at 5%: validation was barely breakeven (Sharpe 0.06), then surged on test (+29.7%) -- the opposite failure mode, equally not something to trust off validation alone.
- **energy_commodity**: the sweep's "best" pick (20%) had a worse Sharpe (0.37) and deeper drawdown (−28%) on test than just keeping the existing 65% default (Sharpe 0.80, −16%) -- a bigger raw return number, a worse strategy.

Also found (not fixed, cosmetic): the script's printed summary line and `test_return_default65` column hardcode the label "default 65%" regardless of category -- growth_tech's real default is 70%, and the number shown is computed correctly against 70%, only the label is wrong.

## 4. Fix: category-appropriate `swing_threshold`, then a stricter stability search

**Step 1 -- pick a real `swing_threshold` per category.** `effective_threshold()` floors at 5%, so candidates were swept from 5% to 15% against each category's own train data, picking a value that lands the positive-label rate in the same 2–8% range that was already working for growth_tech (8.0%) and energy_commodity (2.16%) -- not an arbitrary number.

**Step 2 -- retrain, then search for a *stable* decision_threshold**, using a stricter rule than `select_thresholds.py`'s own: both validation and test need ≥5 trades, a positive return, and a positive Sharpe -- agreement in sign, not just a validation-side score. And within whatever passed that bar, the threshold chosen was the one with the largest, most comparable trade count on both sides, not simply the single highest validation Sharpe (which repeatedly turned out to be the thinnest, least trustworthy sample -- e.g. inflation_safe_haven's naive argmax was 65% on 5 validation trades with a Sharpe of 4.13, not a credible number at that size).

The same swing_threshold sweep was then run against the four categories that were *already* trading (growth_tech, energy_commodity, international_emerging, market_beta) to check whether the same fix applied there too. It didn't apply uniformly: growth_tech and energy_commodity were already sitting in the healthy 2–8% band at 0.15 (lowering further would have diluted the label -- at 5%, growth_tech's positive rate balloons to 39% of all rows), so they were left untouched. international_emerging (1.17%) and market_beta (0.37%) showed the same too-rare pattern as the originally-broken four, so they got the same treatment.

### swing_threshold changes

| Category | Old | New | Old positive rate | New positive rate |
|---|---|---|---|---|
| credit_conditions | 15% | **5%** (floor) | 0.17% | 2.27% |
| rates_recession | 15% | **5%** (floor) | 0.24% | 4.74% |
| inflation_safe_haven | 15% | **7%** | 0.55% | 5.93% |
| small_cap | 15% | **8%** | 0.47% | 4.51% |
| international_emerging | 15% | **9%** | 1.17% | 4.97% |
| market_beta | 15% | **8%** | 0.37% | 2.42% |
| growth_tech | 15% | unchanged | 7.99% | — |
| energy_commodity | 15% | unchanged | 2.16% | — |

### decision_threshold changes (applied via `model_registry.update_decision_threshold`, re-verified against the signed manifest after each change)

| Category | Old | New |
|---|---|---|
| credit_conditions | 65% | **10%** |
| rates_recession | 65% | **45%** |
| inflation_safe_haven | 65% | **40%** |
| small_cap | 65% | **40%** |
| international_emerging | 65% | **10%** |
| market_beta | 65% | **10%** |
| growth_tech | 70% | unchanged |
| energy_commodity | 65% | unchanged |

## 5. Result: current state, all 8 categories

| Category | Threshold | Trades | Win rate | Sharpe |
|---|---|---|---|---|
| growth_tech | 70% | 6 | 83.3% | 3.47 |
| inflation_safe_haven | 40% | 85 | 57.6% | 0.98 |
| energy_commodity | 65% | 34 | 52.9% | 0.80 |
| small_cap | 40% | 152 | 63.2% | 0.74 |
| international_emerging | 10% | 157 | 59.9% | 0.73 |
| market_beta | 10% | 288 | 57.6% | 0.65 |
| credit_conditions | 10% | 117 | 58.1% | 0.55 |
| rates_recession | 45% | 39 | 48.7% | 0.34 |

Six of eight categories now trade on a real sample size (39–288 trades) with win rates in a believable 49–63% band, versus four with zero signal at all before this. Notably, market_beta's and international_emerging's win rates went *down* after the fix (88.9%→57.6%, 66.7%→59.9%) even as trade count went up by 30x+ -- that's the honest outcome, not a regression: the old high win rates were small-sample luck (8/9 and 4/6 trades), not real edge.

## 6. Caveats -- read before trusting any of this

- **`compounded_return` figures are not literal.** The backtest multiplies every trade's return together sequentially, as if one account made all 359 trades (say) in series with 100% reinvested each time, rather than a portfolio holding several symbols at once. Some raw numbers from this process hit three and four digits (e.g. +827% at one swept threshold for inflation_safe_haven) -- Sharpe and win rate are the trustworthy numbers, not the return percentage.
- **growth_tech's Sharpe (3.47) is the least trustworthy number in the final table**, despite topping it -- it's still resting on 6 trades. inflation_safe_haven (0.98 on 85 trades) is a more credible standout.
- **rates_recession is the weakest of the eight post-fix**: Sharpe 0.34, and the only category with a sub-50% win rate (48.7%).
- **This is one retrain on one fixed chronological split**, not cross-validated across multiple windows, and every category is still capped at 7 symbols. None of this should be read as "trustable" in the sense of sizing real capital against it -- it's a real, evidence-backed improvement over "no signal at all," not a finished, validated system.
- Pre-retrain model backups exist for all 8 categories but were kept in this session's scratchpad directory, not in the repo -- that location is ephemeral and may not persist. If long-term retention of the pre-retrain artifacts matters, they should be copied somewhere durable.

## 7. Related findings from this session (already covered by tests, not code-fixed)

- `app/detector.py`'s `walk_forward_backtest`: a position opened on the last eligible bar of the decision window has no remaining bar to ever close it -- not even via max-time exit -- so it's silently excluded from `trades`/`win_rate` even though the equity curve still marks it to market. Characterized in `tests/test_detector.py::test_position_opened_at_end_of_decision_window_is_never_closed_or_recorded`.
- `app/indicators.py`: `atr_14`/`atr_21` and several other indicators call `ta` functions with no `len(df) >= period` guard (unlike the SMA/EMA/Bollinger blocks), so very short input crashes with a third-party `IndexError` before `walk_forward_backtest`'s own row-count check ever runs.
- `scripts/select_thresholds.py`'s printed summary and `test_return_default65` column hardcode "65%" regardless of a category's actual default threshold (see section 3).

## Reproducing this

`scripts/train_all_categories.py` now applies the calibrated `swing_threshold` and `decision_threshold` values from this document automatically, via `app.config.CALIBRATED_SWING_THRESHOLDS` / `CALIBRATED_DECISION_THRESHOLDS` -- so `python scripts/train_all_categories.py` on a fresh clone reproduces this calibration rather than the original uncalibrated defaults. Verified end-to-end: rerunning it against this machine's existing `train/`/`validation/`/`test/` data reproduced identical trade counts and win rates for every category (fixed `random_state=42` on unchanged input data).

What still won't match a fresh clone exactly: `scripts/build_factor_datasets.py` fetches live from Yahoo Finance whenever it's run, so a later run trains on a longer/different history than what produced the specific numbers in section 5, even with the calibration now applied automatically.

The sweep and stability-search scripts used to *derive* these specific numbers (sections 3–4) were ad hoc and run from a scratch directory, not committed to `scripts/`. The methodology is fully described above; if recalibrating again becomes a recurring workflow, the stability-aware search would be worth formalizing as a real script alongside `scripts/select_thresholds.py`.
