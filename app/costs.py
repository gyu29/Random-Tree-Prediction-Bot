"""What a round trip costs, estimated from the price data already on disk.

Every return this project has ever reported is gross. That is a real omission rather than
a rounding one: the separations the significance tests weigh run +0.25% to +1.70% per
trade, and a round trip in the thinner corners of the universe plausibly costs more than
the largest of them. A gross edge that vanishes at the point of trading is not an edge,
and no amount of additional data rescues one -- it would only measure a negative number
more precisely.

There are no bid/ask quotes here, only OHLCV bars, so the spread is estimated from them
using Corwin and Schultz (2012). Their insight is that a day's high-low range reflects
both true volatility and the spread, but volatility scales with the length of the
interval while the spread does not: comparing one two-day range against two consecutive
one-day ranges therefore separates them. It needs nothing but daily highs and lows.

The estimator is noisy day to day and routinely returns negative values for individual
pairs, which the authors address by flooring at zero before averaging. It is a proxy, not
a quote: it captures the order of magnitude and the ranking between symbols, which is
what deciding whether an edge survives trading requires.

Slippage and commission are not modelled separately. For a marketable order in a liquid
ETF the spread dominates; for the illiquid ones the spread estimate is itself an
understatement, since it says nothing about the size available at that price.
"""
import numpy as np
import pandas as pd

# Corwin-Schultz constant: 3 - 2*sqrt(2).
_K = 3 - 2 * np.sqrt(2)

# Floor on the estimate. Even the most liquid ETF costs something to cross, and a
# zero-cost assumption is the one this module exists to remove.
MINIMUM_ROUND_TRIP = 0.0002  # 2 basis points

# Ceiling, to stop a pathological estimate from a near-untraded symbol dominating. A
# symbol whose spread genuinely exceeds this is not tradeable, and the right answer is to
# remove it from the universe rather than to charge the strategy an enormous fee.
MAXIMUM_ROUND_TRIP = 0.05  # 5%


def _column(df, name):
    """High/low by name, case-insensitively. Frames reach this module both normalized by
    app/data_loader.py and straight off disk, where yfinance capitalizes them."""
    for candidate in df.columns:
        if str(candidate).strip().lower() == name:
            return df[candidate]
    raise KeyError(f"no {name!r} column in {list(df.columns)}")


def corwin_schultz_spread(df, price_col=None):
    """Estimated proportional bid-ask spread, as a fraction of price.

    Returns the mean of the daily two-day estimates, negatives floored at zero as the
    authors prescribe. NaN when the frame is too short to form a single pair.
    """
    if len(df) < 2:
        return float("nan")
    high = pd.to_numeric(_column(df, "high"), errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(_column(df, "low"), errors="coerce").to_numpy(dtype=float)
    usable = (high > 0) & (low > 0) & np.isfinite(high) & np.isfinite(low)
    high, low = high[usable], low[usable]
    if len(high) < 2:
        return float("nan")

    log_range = np.log(high / low) ** 2
    beta = log_range[:-1] + log_range[1:]
    two_day_high = np.maximum(high[:-1], high[1:])
    two_day_low = np.minimum(low[:-1], low[1:])
    gamma = np.log(two_day_high / two_day_low) ** 2

    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / _K - np.sqrt(gamma / _K)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    spread = np.where(np.isfinite(spread), spread, np.nan)
    spread = np.clip(spread, 0.0, None)  # negative daily estimates are noise, not credit
    if np.all(np.isnan(spread)):
        return float("nan")
    return float(np.nanmean(spread))


def round_trip_cost(df, price_col=None):
    """Cost of entering and exiting one position, as a fraction of the entry price.

    One full spread: half paid crossing in, half crossing out. Clamped to the range in
    MINIMUM_ROUND_TRIP..MAXIMUM_ROUND_TRIP, and falling back to the minimum when the
    frame cannot support an estimate at all.
    """
    spread = corwin_schultz_spread(df, price_col=price_col)
    if not np.isfinite(spread):
        return MINIMUM_ROUND_TRIP
    return float(np.clip(spread, MINIMUM_ROUND_TRIP, MAXIMUM_ROUND_TRIP))
