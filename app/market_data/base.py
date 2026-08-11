"""Shared contract + shared plumbing for market data providers.

KRXProvider and AlphaVantageProvider talk to genuinely different APIs (KRX is
XML/JSON over a paginated government endpoint with symbol-type fallback;
Alpha Vantage is a single JSON time-series endpoint), so this base class
doesn't try to force a single request-building implementation on both. What
*was* duplicated (and is now shared here) is: request pacing so a market's
rate limit isn't tripped by back-to-back calls, and the final shape every
caller expects back -- a DataFrame with a sorted DatetimeIndex, an
`adj_close` column, and no duplicate dates.
"""
import time
from abc import ABC, abstractmethod

import requests


class MarketDataProvider(ABC):
    def __init__(self, min_request_interval_seconds=0.0):
        self._min_request_interval_seconds = min_request_interval_seconds
        self._last_request_ts = 0.0

    def _paced_get(self, url, params, timeout=30):
        """HTTP GET that never fires sooner than min_request_interval_seconds after the last one."""
        elapsed = time.time() - self._last_request_ts
        if elapsed < self._min_request_interval_seconds:
            time.sleep(self._min_request_interval_seconds - elapsed)
        response = requests.get(url, params=params, timeout=timeout)
        self._last_request_ts = time.time()
        return response

    @staticmethod
    def finalize_ohlcv_frame(df, price_col="close"):
        """Sort, de-duplicate, and ensure adj_close exists. Returns None for an empty frame."""
        if df is None or df.empty:
            return None
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        if "adj_close" not in df.columns:
            df["adj_close"] = df[price_col]
        return df

    @abstractmethod
    def fetch_ohlcv(self, symbol, **kwargs):
        """Returns a DataFrame with open/high/low/close/adj_close/volume, or None."""
        raise NotImplementedError

    @abstractmethod
    def resolve_company_name(self, symbol):
        """Best-effort human-readable name for a symbol; falls back to the symbol itself."""
        raise NotImplementedError
