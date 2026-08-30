"""Coverage for market-context features: causality, alignment, and the missing-data gate.

These are the only features in the project computed from series other than the symbol
being scored, which creates two failure modes nothing else has:

  * Look-ahead through the join. The context series keep a different calendar from any
    given symbol, so they have to be aligned onto the symbol's dates. Aligning with a
    back-fill -- the obvious thing, and what the training path used to do for indicator
    warm-up -- writes a later reading into an earlier row. These tests pin that a row
    dated d sees context through d and no further.
  * Silent substitution at inference. A model trained with context and scored without it
    does not fail; the columns are simply absent, and the detector's missing-feature
    default fills them with zeros, handing the model a market regime that never existed
    and returning a confident number for it. resolve_market_context must raise instead.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.detector import resolve_market_context  # noqa: E402
from app.market_context import (  # noqa: E402
    MARKET_CONTEXT_FEATURES,
    MarketContextUnavailable,
    attach_market_context,
    context_features,
)

DATES = pd.bdate_range("2015-01-01", periods=600, tz="UTC")


def _context(dates=DATES, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "vix": 15 + np.cumsum(rng.normal(0, 0.3, len(dates))).clip(-8, 40),
            "benchmark": 200 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(dates)))),
            "long_rate": 2.5 + np.cumsum(rng.normal(0, 0.01, len(dates))),
            "short_rate": 1.0 + np.cumsum(rng.normal(0, 0.01, len(dates))),
        },
        index=dates,
    )


def _symbol(dates=DATES, seed=1):
    rng = np.random.default_rng(seed)
    close = 50 * np.exp(np.cumsum(rng.normal(0.0004, 0.013, len(dates))))
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "adj_close": close,
         "volume": rng.integers(1_000_000, 3_000_000, len(dates))},
        index=dates,
    )


def test_every_declared_feature_is_produced():
    attached = attach_market_context(_symbol(), _context())
    missing = [name for name in MARKET_CONTEXT_FEATURES if name not in attached.columns]
    assert missing == [], f"declared but not produced: {missing}"


def test_attaching_context_preserves_rows_and_original_columns():
    symbol = _symbol()
    attached = attach_market_context(symbol, _context())
    assert len(attached) == len(symbol)
    assert attached.index.equals(symbol.index)
    for column in symbol.columns:
        assert np.allclose(attached[column], symbol[column])


def test_context_features_do_not_use_future_information():
    """Truncating the context after date d must not change any row at or before d. If a
    later reading is reachable from an earlier row, this fails."""
    symbol = _symbol()
    context = _context()
    cutoff = DATES[400]

    full = attach_market_context(symbol, context)
    truncated = attach_market_context(symbol[symbol.index <= cutoff], context[context.index <= cutoff])

    overlap = truncated.index
    for feature in MARKET_CONTEXT_FEATURES:
        a = full.loc[overlap, feature].to_numpy()
        b = truncated[feature].to_numpy()
        both_present = ~(np.isnan(a) | np.isnan(b))
        assert np.allclose(a[both_present], b[both_present]), f"{feature} changed when the future was removed"


def test_regime_features_do_not_use_future_information():
    context = _context()
    cutoff = DATES[300]
    full = context_features(context)
    truncated = context_features(context[context.index <= cutoff])
    for column in truncated.columns:
        a = full.loc[truncated.index, column].to_numpy()
        b = truncated[column].to_numpy()
        both = ~(np.isnan(a) | np.isnan(b))
        assert np.allclose(a[both], b[both]), f"{column} changed when the future was removed"


def test_a_gap_in_the_context_carries_forward_not_backward():
    """A symbol trading on a day the context series did not print gets the last known
    reading, never the next one."""
    context = _context()
    holiday = DATES[200]
    with_gap = context.drop(index=holiday)

    attached = attach_market_context(_symbol(), with_gap)
    previous = context_features(with_gap)["vix_level"].asof(DATES[199])
    assert attached.loc[holiday, "vix_level"] == pytest.approx(previous)


def test_leading_rows_without_context_are_left_missing():
    """Before the context series begin there is nothing to carry forward. Those rows must
    stay NaN so the training path drops them, rather than being filled from later data --
    ^VIX starts in 1990 and the oldest symbol here reaches back to 1927."""
    context = _context()
    late_context = context[context.index >= DATES[100]]
    attached = attach_market_context(_symbol(), late_context)
    assert attached["vix_level"].iloc[:100].isna().all()
    assert attached["vix_level"].iloc[150:].notna().any()


def test_excess_return_is_the_symbol_minus_the_benchmark():
    context = _context()
    symbol = _symbol()
    attached = attach_market_context(symbol, context)
    expected = (
        symbol["adj_close"].pct_change(5).to_numpy()
        - context["benchmark"].reindex(symbol.index).pct_change(5).to_numpy()
    )
    actual = attached["excess_return_5"].to_numpy()
    both = ~(np.isnan(expected) | np.isnan(actual))
    assert np.allclose(expected[both], actual[both])


def test_missing_context_is_an_error_not_an_empty_frame():
    with pytest.raises(MarketContextUnavailable):
        attach_market_context(_symbol(), None)


# -- the inference gate ---------------------------------------------------------------


class _Detector:
    def __init__(self, uses_market_context=False, market_context=None):
        self.category = "widgets"
        self.uses_market_context = uses_market_context
        self.market_context = market_context


def test_a_model_that_needs_context_and_lacks_it_raises():
    """Not returning None: the columns would go missing, the detector's missing-feature
    default would fill them with zeros, and the model would score a regime that never
    existed without anything looking wrong."""
    with pytest.raises(MarketContextUnavailable):
        resolve_market_context(_Detector(uses_market_context=True, market_context=None))


def test_a_model_that_needs_context_and_has_it_gets_it():
    context = _context()
    resolved = resolve_market_context(_Detector(uses_market_context=True, market_context=context))
    assert resolved is context


def test_a_model_trained_without_context_needs_none():
    assert resolve_market_context(_Detector(uses_market_context=False)) is None


def test_a_stand_in_that_declares_nothing_is_treated_as_context_free():
    """walk_forward_backtest is duck-typed and driven by stand-ins in tests and by
    FoldDetector; one that declares nothing must not trip the gate."""
    class _Bare:
        pass

    assert resolve_market_context(_Bare()) is None


# -- short frames ---------------------------------------------------------------------


@pytest.mark.parametrize("rows", [1, 5, 12, 14, 20, 26, 27, 28, 40, 61])
def test_indicators_never_raise_on_a_short_frame(rows):
    """`ta`'s adx indexes its output at the window length without checking the array is
    that long, so a frame shorter than its lookback used to raise IndexError from inside
    the library -- crashing three categories of cross-validation on a 12-row fold slice.
    Short input must produce missing columns, which callers drop, not an exception."""
    from app.indicators import TechnicalIndicators

    rng = np.random.default_rng(rows)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, rows)))
    frame = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
         "adj_close": close, "volume": rng.integers(1_000_000, 2_000_000, rows)},
        index=pd.bdate_range("2020-01-01", periods=rows, tz="UTC"),
    )
    for context in (None, _context(dates=frame.index)):
        result = TechnicalIndicators.create_all_indicators(frame, market_context=context)
        assert len(result) == rows


# -- the backtest window must not be seeded from the future ---------------------------


def test_backtest_window_starts_only_once_features_exist():
    """score_for_backtest used to ffill *and* bfill, which wrote later bars' values into
    the warm-up rows at the start of the window -- about 10% of every backtest's decision
    bars were scored on features derived from their own future. The window must instead
    begin where the features do."""
    from app.detector import score_for_backtest

    rows = 400
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, rows)))
    frame = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
         "adj_close": close, "volume": rng.integers(1_000_000, 2_000_000, rows)},
        index=pd.bdate_range("2018-01-01", periods=rows, tz="UTC"),
    )

    class _Detector:
        is_ready = True
        category = "widgets"
        lookforward_periods = 10
        decision_threshold = 0.5
        uses_market_context = False
        market_context = None
        # A 200-period feature: unavailable for the first 199 rows by construction.
        feature_columns = ["price_sma_200_ratio", "rsi_14"]

        class scaler:
            @staticmethod
            def transform(X):
                assert not np.isnan(np.asarray(X, dtype=float)).any(), (
                    "a bar with missing features reached the model"
                )
                return np.asarray(X, dtype=float)

        class model:
            @staticmethod
            def predict_proba(X):
                return np.column_stack([np.ones(len(X)) * 0.9, np.ones(len(X)) * 0.1])

    scoring = score_for_backtest(_Detector(), frame)
    assert scoring.decision_start >= 199, (
        "window opened on bars whose 200-period feature could not yet be computed"
    )
    scored = scoring.features[_Detector.feature_columns].iloc[
        scoring.decision_start:scoring.decision_end
    ]
    assert scored.notna().all().all()
