"""Thread-safe shared state for the Qt UI.

The original UI read and wrote self.system.detector / self.watchlist /
self.last_analysis directly from both the main thread (settings changes, market
toggles, training completion callbacks) and QThreadPool worker threads (analysis,
dashboard refresh, backtests) with no lock -- two workers could race to
initialize a detector, or a settings change could interleave with an in-flight
read. market_history_cache was already correctly lock-protected; AppState
generalizes that same pattern (one RLock) to cover the rest of the shared state
so there's exactly one place all of it is touched from.
"""
import threading
import time

from app.data_loader import category_for_symbol
from app.trading_system import SwingTradingSystem


class AppState:
    def __init__(self, api_key=None):
        self._lock = threading.RLock()
        self.system = SwingTradingSystem()
        self._watchlist = []
        self._last_analysis = None
        self._last_backtest = None
        self._market_history_cache = {}

    # -- detector -----------------------------------------------------------
    def get_detector(self, category, stock_mode):
        with self._lock:
            return self.system.get_detector(category, stock_mode)

    def train_model(self, category, **kwargs):
        """Goes through the same lock as get_detector/invalidate_detector: train_model
        mutates the detector cache internally (it calls invalidate_detector on success),
        so it needs the same synchronization as every other path that touches
        self.system.detectors -- calling self.state.system.train_model(...) directly
        from a task would bypass that."""
        with self._lock:
            return self.system.train_model(category, **kwargs)

    def invalidate_detector(self, category=None):
        with self._lock:
            self.system.invalidate_detector(category)

    def ensure_detector(self, symbol, stock_mode, category=None):
        resolved_category = category or category_for_symbol(symbol)
        with self._lock:
            return resolved_category, self.system.get_detector(resolved_category, stock_mode)

    # -- watchlist ------------------------------------------------------------
    # Each entry is {"symbol": str, "category": str | None}. A None category means
    # "no override" -- callers resolve it via category_for_symbol at use time (see
    # ensure_detector above), so entries added before per-symbol categories existed,
    # or left on "(auto-detect)" deliberately, keep tracking FACTOR_CATEGORIES if it
    # ever changes rather than freezing in a guess made at add-time.
    @staticmethod
    def _normalize_watchlist_entry(item):
        if isinstance(item, dict):
            return {"symbol": str(item.get("symbol", "")).upper(), "category": item.get("category") or None}
        return {"symbol": str(item).upper(), "category": None}

    def get_watchlist(self):
        with self._lock:
            return [entry["symbol"] for entry in self._watchlist]

    def get_watchlist_entries(self):
        with self._lock:
            return [dict(entry) for entry in self._watchlist]

    def set_watchlist(self, items):
        """Accepts plain symbol strings (legacy config / stocks.json) or
        {"symbol", "category"} dicts (current trading_ui_config.json format) -- either
        is normalized to the same internal shape, so callers don't need to know which
        format they're loading."""
        with self._lock:
            self._watchlist = [self._normalize_watchlist_entry(item) for item in items]

    def append_watchlist(self, symbol, category=None):
        with self._lock:
            if any(entry["symbol"] == symbol for entry in self._watchlist):
                return False
            self._watchlist.append({"symbol": symbol, "category": category})
            return True

    def remove_watchlist_at(self, index):
        with self._lock:
            if 0 <= index < len(self._watchlist):
                self._watchlist.pop(index)
                return True
            return False

    def set_watchlist_category_at(self, index, category):
        with self._lock:
            if 0 <= index < len(self._watchlist):
                self._watchlist[index]["category"] = category
                return True
            return False

    # -- last analysis / backtest -------------------------------------------
    def get_last_analysis(self):
        with self._lock:
            return self._last_analysis

    def set_last_analysis(self, payload):
        with self._lock:
            self._last_analysis = payload

    def get_last_backtest(self):
        with self._lock:
            return self._last_backtest

    def set_last_backtest(self, payload):
        with self._lock:
            self._last_backtest = payload

    # -- market history cache -------------------------------------------------
    def get_cached_history(self, key, ttl_seconds):
        with self._lock:
            cached = self._market_history_cache.get(key)
            if not cached:
                return None
            cached_at, df = cached
            if time.time() - cached_at >= ttl_seconds:
                self._market_history_cache.pop(key, None)
                return None
            return df.copy()

    def set_cached_history(self, key, df):
        with self._lock:
            self._market_history_cache[key] = (time.time(), df.copy())

    def clear_market_cache(self):
        with self._lock:
            self._market_history_cache.clear()
