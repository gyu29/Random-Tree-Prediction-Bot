"""Market-wide context: regime state and each symbol's standing relative to a benchmark.

Every other feature in this project describes one symbol in isolation, which leaves the
models unable to distinguish two situations that call for opposite decisions: "this name
just became volatile" and "everything just became volatile". It also makes the factor
categories partly incoherent -- small_cap is a claim about small caps *relative to large
caps*, and a model shown only IWM's own bars cannot see the comparison its category is
named after.

Four shared series supply both:

    ^VIX   implied volatility -- the regime everything trades in
    SPY    the benchmark every relative measure is taken against
    ^TNX   10-year Treasury yield
    ^IRX   13-week Treasury yield  (^TNX - ^IRX is the term spread, the standard
                                    recession signal and a broad risk-appetite proxy)

Yields and VIX are already normalized quantities -- percentages, comparable across
decades -- so unlike price they can be used at their level without the extrapolation
problem described in app/indicators.py. Everything derived from SPY is a ratio or a
difference of returns, never a price.

All features are causal: rolling windows and backward differences only, computed on the
context series before the join, so a symbol's row on date d sees context through d and
no further.

The series are cached to market_context.csv (gitignored, rebuilt by
scripts/build_factor_datasets.py) so training and inference read the same numbers, and
so scoring a single symbol in the desktop app does not require refetching four series.
"""
import os

import numpy as np
import pandas as pd

from app.config import MARKET_CONTEXT_PATH

CONTEXT_SERIES = {
    "vix": "^VIX",
    "benchmark": "SPY",
    "long_rate": "^TNX",
    "short_rate": "^IRX",
}

# Columns attach_market_context adds. Named here so app/trainer.py and app/detector.py
# can agree on what "the model was trained with context" means without recomputing.
MARKET_CONTEXT_FEATURES = [
    "vix_level", "vix_change_5", "vix_zscore_60",
    "term_spread", "term_spread_change_20",
    "benchmark_return_5", "benchmark_return_20", "benchmark_volatility_20",
    "excess_return_5", "excess_return_10", "excess_return_20",
    "relative_strength_60", "beta_60",
]


class MarketContextUnavailable(Exception):
    """Raised when a model trained with context features is asked to score without them.
    Filling them with zeros instead would silently feed the model a regime it never saw."""


def fetch_context_series():
    """Downloads the four series. Imports yfinance lazily -- it is only needed when
    building datasets, not when running the app."""
    import yfinance as yf

    frames = {}
    for name, ticker in CONTEXT_SERIES.items():
        history = yf.Ticker(ticker).history(period="max")
        if history.empty:
            raise ValueError(f"no history returned for {ticker}")
        # Normalize each series to a UTC calendar date *before* combining. yfinance
        # returns these with different exchange timezones and session times, so aligning
        # the raw indexes puts each series on its own distinct timestamps and every
        # column ends up NaN wherever another column has data.
        series = history["Close"]
        series.index = pd.to_datetime(series.index, utc=True).normalize()
        frames[name] = series[~series.index.duplicated(keep="last")]
    combined = pd.DataFrame(frames).sort_index()
    # The four series keep slightly different calendars (one holiday, one late close), so
    # the union join leaves gaps. Forward-fill only: carrying yesterday's VIX into a day
    # it hasn't printed yet is what a live reader would see, while a back-fill would put
    # tomorrow's number in today's row.
    return combined.ffill()


def save_context(frame, path=None):
    path = path or MARKET_CONTEXT_PATH
    frame.to_csv(path)
    return path


def load_context(path=None):
    """The cached series, or None when they have never been built."""
    path = path or MARKET_CONTEXT_PATH
    if not os.path.isfile(path):
        return None
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_datetime(frame.index, utc=True).normalize()
    return frame.sort_index()


def context_features(context):
    """Regime columns, derived once and shared by every symbol on a given date."""
    derived = pd.DataFrame(index=context.index)
    vix = context["vix"]
    derived["vix_level"] = vix / 100.0
    derived["vix_change_5"] = vix.pct_change(5)
    rolling = vix.rolling(60, min_periods=60)
    derived["vix_zscore_60"] = (vix - rolling.mean()) / rolling.std().replace(0.0, np.nan)

    spread = context["long_rate"] - context["short_rate"]
    derived["term_spread"] = spread / 100.0
    derived["term_spread_change_20"] = spread.diff(20) / 100.0

    benchmark = context["benchmark"]
    derived["benchmark_return_5"] = benchmark.pct_change(5)
    derived["benchmark_return_20"] = benchmark.pct_change(20)
    derived["benchmark_volatility_20"] = benchmark.pct_change().rolling(20, min_periods=20).std()
    return derived


def attach_market_context(df, context, price_col=None):
    """Adds MARKET_CONTEXT_FEATURES to one symbol's frame.

    The regime columns are a straight date join. The relative columns need the symbol and
    the benchmark side by side, so they are computed after aligning the benchmark onto
    this symbol's dates -- forward-filled, never back-filled, because a back-fill would
    carry a later benchmark price into an earlier row.
    """
    if context is None:
        raise MarketContextUnavailable(
            "market context has not been built -- run scripts/build_factor_datasets.py"
        )
    price_col = price_col or ("adj_close" if "adj_close" in df.columns else "close")
    result = df.copy()

    dates = pd.to_datetime(result.index, utc=True).normalize()
    regime = context_features(context)
    aligned_regime = regime.reindex(regime.index.union(dates)).ffill().reindex(dates)
    for column in aligned_regime.columns:
        result[column] = aligned_regime[column].to_numpy()

    benchmark = context["benchmark"]
    aligned_benchmark = pd.Series(
        benchmark.reindex(benchmark.index.union(dates)).ffill().reindex(dates).to_numpy(),
        index=result.index,
    )
    price = result[price_col]
    for window in (5, 10, 20):
        result[f"excess_return_{window}"] = (
            price.pct_change(window) - aligned_benchmark.pct_change(window)
        )

    ratio = price / aligned_benchmark
    ratio_window = ratio.rolling(60, min_periods=60)
    result["relative_strength_60"] = (
        (ratio - ratio_window.mean()) / ratio_window.std().replace(0.0, np.nan)
    )

    symbol_returns = price.pct_change()
    benchmark_returns = aligned_benchmark.pct_change()
    covariance = symbol_returns.rolling(60, min_periods=60).cov(benchmark_returns)
    variance = benchmark_returns.rolling(60, min_periods=60).var().replace(0.0, np.nan)
    result["beta_60"] = covariance / variance
    return result
