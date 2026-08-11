"""Background-task bodies run on QThreadPool workers (see ui/widgets.py's Worker).

Error-handling rule, applied consistently at every call site below: detector.
NoModelAvailableError is the only condition allowed to fall back to a heuristic
estimate, and that estimate is always tagged source="heuristic_no_model" with a
human-readable reason so the UI can render a visible "not model-based" badge
instead of a result that looks like a real prediction. Anything else --
PredictionError, a network error, a bug -- propagates as a real exception. For
background tasks that means Worker.run() catches it and emits `failed`, which
ui/app_window.py already turns into a QMessageBox.critical -- that path existed
before this rewrite and worked correctly; the bug was tasks swallowing errors
before they could reach it.
"""
import os

import numpy as np
import pandas as pd

from app.config import ALPHA_VANTAGE_CACHE_TTL_SECONDS, DEFAULT_DECISION_THRESHOLD, LIVE_MARKET_CACHE_TTL_SECONDS
from app.data_loader import DataProcessor, category_for_symbol
from app.detector import NoModelAvailableError, walk_forward_backtest
from app.security import SecurityValidator
from ui.widgets import capture_stdout, silence_stdout

_COMPANY_NAMES = {
    "005930": "Samsung Electronics",
    "000660": "SK hynix",
    "035420": "NAVER",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "GOOGL": "Alphabet",
    "GOOG": "Alphabet",
    "AMZN": "Amazon",
    "META": "Meta Platforms",
    "TSLA": "Tesla",
    "JPM": "JPMorgan Chase",
}


def company_name(symbol):
    return _COMPANY_NAMES.get(symbol.upper(), symbol.upper())


def fallback_probability(momentum, rsi, volume_ratio):
    """Simple momentum/RSI/volume heuristic -- ONLY used when detector.NoModelAvailableError
    fires (i.e. nothing has been trained for this category yet). Never used to paper over a
    real prediction failure; see module docstring."""
    momentum_score = max(-0.2, min(0.2, momentum)) * 1.7
    rsi_score = max(-0.1, min(0.16, (50 - abs(float(rsi) - 55)) / 300))
    volume_score = max(-0.1, min(0.15, (float(volume_ratio) - 1) * 0.08))
    return max(0.05, min(0.95, 0.48 + momentum_score + rsi_score + volume_score))


def fallback_analysis(symbol, df, mode, reason):
    latest = df.iloc[-1]
    previous = df.iloc[-2] if len(df) > 1 else latest
    price = float(latest.get("close", 0))
    previous_price = float(previous.get("close", price))
    change = (price - previous_price) / previous_price if previous_price else 0
    recent = df.tail(15)
    start = float(recent.iloc[0]["close"])
    momentum = (float(recent.iloc[-1]["close"]) - start) / start if start else 0
    average_volume = float(df.tail(30)["volume"].mean()) if "volume" in df else 1
    current_volume = float(latest.get("volume", 0))
    volume_ratio = current_volume / average_volume if average_volume else 1
    gains = df["close"].diff().clip(lower=0).tail(14).mean()
    losses = (-df["close"].diff().clip(upper=0)).tail(14).mean()
    rsi = 100 - (100 / (1 + gains / losses)) if losses else 50
    probability = fallback_probability(momentum, rsi, volume_ratio)
    swing = 0.08 if mode == "US" else 0.05
    return {
        "symbol": symbol,
        "current_price": price,
        "current_volume": current_volume,
        "price_change_1d": change,
        "swing_probability": probability,
        "is_swing_opportunity": probability >= 0.5,
        "confidence_level": "High" if probability >= 0.78 else "Medium" if probability >= 0.6 else "Low",
        "stop_loss": price * (1 - swing * 0.5),
        "take_profit": price * (1 + swing),
        "currency": "KRW" if mode == "KR" else "USD",
        "source": "heuristic_no_model",
        "heuristic_reason": reason,
    }


class TaskController:
    """Wraps AppState with the actual task bodies Worker instances run."""

    def __init__(self, state):
        self.state = state

    # -- market data ----------------------------------------------------------
    def fetch_market_history(self, symbol, mode, outputsize="compact"):
        normalized = SecurityValidator.validate_symbol(symbol, mode, self.state.system.security_config)
        key = (mode, normalized, outputsize)
        ttl = ALPHA_VANTAGE_CACHE_TTL_SECONDS if mode == "US" else LIVE_MARKET_CACHE_TTL_SECONDS
        cached = self.state.get_cached_history(key, ttl)
        if cached is not None:
            return cached

        _category, detector = self.state.ensure_detector(normalized, mode)
        df = detector.fetch_market_data(normalized, outputsize=outputsize)
        if df is None or df.empty:
            return None
        df = df.sort_index()
        self.state.set_cached_history(key, df)
        return df.copy()

    # -- single-symbol analysis -------------------------------------------------
    def analysis_task(self, symbol, mode, category=None):
        with capture_stdout() as log:
            resolved_category, detector = self.state.ensure_detector(symbol, mode, category)
            df = self.fetch_market_history(symbol, mode)
            if df is None or len(df) < 50:
                raise ValueError(f"Could not fetch enough market data for {symbol}.")

            try:
                result = detector.detect_swing_opportunity(df, symbol)
            except NoModelAvailableError:
                result = fallback_analysis(
                    symbol, df, mode, f"No trained model yet for category '{resolved_category}'."
                )
            result["currency"] = result.get("currency") or ("KRW" if mode == "KR" else "USD")
            result["category"] = resolved_category

        return {"result": result, "df": df, "company": company_name(symbol), "log": log.getvalue()}

    # -- dashboard / screener / monitor -----------------------------------------
    def compute_dashboard_data(self, watchlist_entries, mode):
        signals = []
        watch_items = []
        displayed_dates = []
        for entry in watchlist_entries:
            symbol = entry["symbol"]
            category_override = entry.get("category")
            try:
                with silence_stdout():
                    resolved_category, detector = self.state.ensure_detector(symbol, mode, category_override)
                    df = self.fetch_market_history(symbol, mode)
                if df is None or len(df) < 2:
                    raise ValueError("No live data")
                as_of = _date_from_index(df.index[-1])
                if as_of:
                    displayed_dates.append(as_of)
                price = float(df.iloc[-1]["close"])
                previous = float(df.iloc[-2]["close"])
                change = (price - previous) / previous if previous else 0
                watch_items.append({
                    "symbol": symbol, "price": price, "change": change,
                    "currency": "KRW" if mode == "KR" else "USD",
                })

                with silence_stdout():
                    if len(df) >= 100:
                        try:
                            result = detector.detect_swing_opportunity(df, symbol)
                        except NoModelAvailableError:
                            result = fallback_analysis(
                                symbol, df, mode, f"No trained model yet for category '{resolved_category}'."
                            )
                    else:
                        result = fallback_analysis(symbol, df, mode, "Fewer than 100 rows of history available.")

                signals.append({
                    "symbol": symbol,
                    "company": company_name(symbol),
                    "category": resolved_category,
                    "probability": float(result.get("swing_probability", 0)),
                    "confidence": result.get("confidence_level", "Low"),
                    "source": result.get("source", "model"),
                })
            except Exception as error:
                # A real failure for this symbol (bad data, PredictionError, no API key, ...).
                # Reported as unavailable, never disguised as a probability. See module docstring.
                # category_override or category_for_symbol (not resolved_category) here: the
                # failure may have happened inside ensure_detector itself, before resolved_category
                # was ever assigned -- but the user's explicit override, if set, is still known.
                signals.append({
                    "symbol": symbol, "company": company_name(symbol),
                    "category": category_override or category_for_symbol(symbol), "probability": 0,
                    "confidence": "Unavailable", "status": "unavailable", "message": str(error),
                })

        signals.sort(key=lambda item: (item.get("status") == "unavailable", -item.get("probability", 0)))
        available = [item["probability"] for item in signals if item.get("status") != "unavailable"]

        return {
            "signals": signals,
            "watchlist": watch_items,
            "avg_probability": float(np.mean(available)) if available else 0,
            "signals_today": len([value for value in available if value >= DEFAULT_DECISION_THRESHOLD]),
            "displayed_dates": displayed_dates,
        }

    # -- training -----------------------------------------------------------------
    def training_task(self, category, parameters):
        with capture_stdout() as log:
            result = self.state.train_model(
                category,
                swing_threshold=parameters["swing_threshold"],
                lookforward_periods=parameters["swing_window"],
                rf_estimators=parameters["rf_estimators"],
                xgb_learning_rate=parameters["learning_rate"],
                xgb_max_depth=parameters["max_depth"],
            )
        return {
            "category": category,
            "validation_score": result["validation_score"],
            "training_stats": result["training_stats"],
            "log": log.getvalue(),
        }

    # -- backtest -------------------------------------------------------------------
    def backtest_task(self, category, symbol, mode, days_back, threshold, file_path, df=None):
        """category is an explicit model choice (which trained category's detector to
        score file_path with), not inferred from the symbol -- the caller decides both
        independently.

        df: optionally a DataFrame the caller already parsed from file_path (see
        ui/pages/backtest.py's _load_df_cached), so a file picked once and backtested
        several times over doesn't get re-read from disk on every run. Falls back to
        parsing file_path itself when not given, so this stays correct if ever called
        without a pre-parsed frame."""
        with silence_stdout():
            _resolved_category, detector = self.state.ensure_detector(symbol, mode, category)
            if df is None:
                df = DataProcessor.load_and_validate_data(file_path)
            if df is None or df.empty:
                raise ValueError(f"Not enough usable history in {os.path.basename(file_path)}.")

            cutoff = df.index.max() - pd.Timedelta(days=days_back)
            df_recent = df[df.index >= cutoff].copy()

        result = walk_forward_backtest(detector, df_recent, decision_threshold=threshold)
        result["symbol"] = symbol
        result["category"] = category
        result["file_path"] = file_path
        return result


def _date_from_index(value):
    try:
        timestamp = pd.to_datetime(value, errors="coerce", utc=True)
        return None if pd.isna(timestamp) else timestamp.date()
    except (TypeError, ValueError):
        return None
