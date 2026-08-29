"""Public-facing orchestration: training, analysis, monitoring, screening, backtesting.

Validation and rate-limit failures (SecurityValidationError, RateLimitExceededError)
propagate as real exceptions rather than being converted into a dict shaped like a
success response -- a caller that doesn't explicitly check for an error shape would
otherwise misread a rate-limited call as having succeeded (this bit the very first
real run of scripts/train_all_categories.py: category 4 hit training_limit_per_user_hour
and the caller's `result["training_stats"]` blew up with an opaque KeyError instead of
the actual RateLimitExceededError). Callers that need a structured response instead of
an exception (e.g. a future HTTP layer) can catch these and call error.to_response() /
str(error) themselves.

Not thread-safe on its own -- a single SwingTradingSystem's detector cache is a plain
dict. Callers that use this from multiple threads (the Qt UI) are responsible for
synchronizing access; see ui/state.py's AppState, which is the only place that should
hold a SwingTradingSystem shared across QThreadPool workers.
"""
import glob
import hashlib
import json
import os

import pandas as pd

from app import model_registry
from app.config import STOCKS_FILE_PATH
from app.data_loader import category_for_symbol, category_train_dir, list_categories
from app.detector import SwingTradeDetector, walk_forward_backtest
from app.market_data import AlphaVantageProvider, KRXProvider
from app.security import (
    RequestRateLimiter,
    SecretManager,
    SecurityConfig,
    SecurityValidationError,
    SecurityValidator,
)
from app.trainer import SwingTradeTrainer


def _hash_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset_snapshot(data_directory):
    snapshot = []
    for path in sorted(glob.glob(os.path.join(data_directory, "*.csv"))):
        df = pd.read_csv(path, usecols=[0])
        date_col = df.iloc[:, 0]
        snapshot.append({
            "file": os.path.basename(path),
            "sha256": _hash_file(path),
            "rows": int(len(df)),
            "date_min": str(date_col.min()),
            "date_max": str(date_col.max()),
        })
    return snapshot


def _screen_sort_key(entry):
    """Sort key for screen_symbols: (rankable, probability), descending. Only status="ok"
    rows are rankable; a row whose model failed validation ("untrusted_model") or whose
    data never arrived ("unavailable") sorts below every ranked one, without its
    probability ever being compared against a trustworthy one."""
    rankable = 1 if entry.get("status") == "ok" else 0
    return (rankable, entry.get("swing_probability", 0))


class SwingTradingSystem:
    def __init__(self):
        self.security_config = SecurityConfig()
        self.rate_limiter = RequestRateLimiter()
        self.detectors = {}  # category -> (stock_mode, SwingTradeDetector)

    def _enforce_public_rate_limit(self, operation, request_context, ip_limit, user_limit, window_seconds):
        validated_context = SecurityValidator.validate_request_context(request_context, self.security_config)
        self.rate_limiter.enforce(
            operation, validated_context["ip_address"], validated_context["username"],
            ip_limit, user_limit, window_seconds,
        )

    def _build_provider(self, stock_mode):
        if stock_mode == "US":
            api_key = SecretManager.get_optional_secret("ALPHA_VANTAGE_API_KEY")
            if not api_key:
                raise ValueError("ALPHA_VANTAGE_API_KEY is required for live US market data.")
            return AlphaVantageProvider(api_key=api_key)
        return KRXProvider()

    def get_detector(self, category, stock_mode="US"):
        cached = self.detectors.get(category)
        if cached is not None and cached[0] == stock_mode:
            return cached[1]
        detector = SwingTradeDetector(category, self._build_provider(stock_mode))
        self.detectors[category] = (stock_mode, detector)
        return detector

    def invalidate_detector(self, category=None):
        if category is None:
            self.detectors.clear()
        else:
            self.detectors.pop(category, None)

    def train_model(
        self,
        category,
        data_directory=None,
        swing_threshold=0.15,
        lookforward_periods=10,
        rf_estimators=250,
        xgb_learning_rate=0.05,
        xgb_max_depth=6,
        request_context=None,
    ):
        self._enforce_public_rate_limit(
            "train_model", request_context,
            self.security_config.training_limit_per_hour, self.security_config.training_limit_per_user_hour, 3600,
        )
        if category not in list_categories():
            raise SecurityValidationError(f"Unknown category: {category}")
        data_directory = data_directory or category_train_dir(category)
        validated_data_directory = SecurityValidator.validate_data_directory(data_directory, self.security_config)
        validated_threshold = SecurityValidator.validate_float(swing_threshold, "swing_threshold", 0.01, 1.0)
        validated_lookforward = SecurityValidator.validate_int(lookforward_periods, "lookforward_periods", 2, 90)
        # Bounds match the Train model UI's spinbox ranges (ui/pages/train.py) so a value the
        # UI allows the user to pick can never be rejected here.
        validated_rf_estimators = SecurityValidator.validate_int(rf_estimators, "rf_estimators", 50, 500)
        validated_xgb_learning_rate = SecurityValidator.validate_float(xgb_learning_rate, "xgb_learning_rate", 0.01, 0.20)
        validated_xgb_max_depth = SecurityValidator.validate_int(xgb_max_depth, "xgb_max_depth", 3, 10)

        trainer = SwingTradeTrainer(
            swing_threshold=validated_threshold,
            lookforward_periods=validated_lookforward,
            rf_estimators=validated_rf_estimators,
            xgb_learning_rate=validated_xgb_learning_rate,
            xgb_max_depth=validated_xgb_max_depth,
        )
        validation_score = trainer.train(validated_data_directory)
        dataset_snapshot = build_dataset_snapshot(validated_data_directory)
        model_registry.save(
            category,
            model=trainer.model,
            scaler=trainer.scaler,
            feature_columns=trainer.feature_columns,
            training_stats=trainer.training_stats,
            dataset_snapshot=dataset_snapshot,
            hyperparameters=trainer.hyperparameters(),
        )
        self.invalidate_detector(category)
        return {"validation_score": validation_score, "training_stats": trainer.training_stats, "category": category}

    def analyze_symbol(self, symbol, stock_mode="US", category=None, request_context=None):
        self._enforce_public_rate_limit(
            "analyze_symbol", request_context,
            self.security_config.analysis_limit_per_minute, self.security_config.analysis_limit_per_user_minute, 60,
        )
        validated_mode = SecurityValidator.validate_choice(stock_mode, "stock_mode", {"US", "KR"})
        validated_symbol = SecurityValidator.validate_symbol(symbol, validated_mode, self.security_config)

        resolved_category = category or category_for_symbol(validated_symbol)
        detector = self.get_detector(resolved_category, validated_mode)
        df = detector.fetch_market_data(validated_symbol)
        if df is None or len(df) < 50:
            raise ValueError(f"Could not fetch enough market data for {validated_symbol}.")
        return detector.detect_swing_opportunity(df, validated_symbol)

    def monitor_symbols(self, symbols, check_interval=300, alert_threshold=0.7, stock_mode="US", category=None, request_context=None):
        self._enforce_public_rate_limit(
            "monitor_symbols", request_context,
            self.security_config.monitor_limit_per_minute, self.security_config.monitor_limit_per_user_minute, 60,
        )
        validated_mode = SecurityValidator.validate_choice(stock_mode, "stock_mode", {"US", "KR"})
        validated_symbols = SecurityValidator.validate_symbol_list(symbols, validated_mode, self.security_config)
        validated_threshold = SecurityValidator.validate_float(alert_threshold, "alert_threshold", 0.0, 1.0)
        validated_interval = SecurityValidator.validate_int(check_interval, "check_interval", 5, 86400)

        results = {}
        suppressed = {}
        for symbol in validated_symbols:
            resolved_category = category or category_for_symbol(symbol)
            detector = self.get_detector(resolved_category, validated_mode)
            df = detector.fetch_market_data(symbol)
            if df is None or len(df) < 50:
                continue
            result = detector.detect_swing_opportunity(df, symbol)
            # An alert is a recommendation to act. A category whose model lost to its own
            # null out-of-sample (CATEGORIES_FAILING_VALIDATION) never raises one, however
            # high its probability -- the probability is the thing that isn't meaningful.
            if not detector.is_trustworthy:
                suppressed[symbol] = result
                continue
            if result["swing_probability"] >= validated_threshold:
                results[symbol] = result
        return {"alerts": results, "suppressed": suppressed, "check_interval": validated_interval}

    def load_symbols_from_stocks_file(self, stock_mode="US", stocks_file=STOCKS_FILE_PATH):
        if not os.path.isfile(stocks_file):
            return []
        with open(stocks_file, "r", encoding="utf-8") as handle:
            symbols = json.load(handle)
        return SecurityValidator.validate_symbol_list(symbols, stock_mode, self.security_config)

    def screen_symbols(self, symbols, stock_mode="US", category=None, request_context=None):
        self._enforce_public_rate_limit(
            "screen_symbols", request_context,
            self.security_config.analysis_limit_per_minute, self.security_config.analysis_limit_per_user_minute, 60,
        )
        validated_mode = SecurityValidator.validate_choice(stock_mode, "stock_mode", {"US", "KR"})
        validated_symbols = SecurityValidator.validate_symbol_list(symbols, validated_mode, self.security_config)

        screened = []
        for symbol in validated_symbols:
            resolved_category = category or category_for_symbol(symbol)
            detector = self.get_detector(resolved_category, validated_mode)
            df = detector.fetch_market_data(symbol)
            if df is None or len(df) < 50:
                screened.append({"symbol": symbol, "status": "unavailable"})
                continue
            result = detector.detect_swing_opportunity(df, symbol)
            result["status"] = "ok" if detector.is_trustworthy else "untrusted_model"
            screened.append(result)
        # A ranking is a recommendation, so a model that failed its out-of-sample check
        # doesn't get a rank: its rows sort below every scored one regardless of
        # probability, and carry status="untrusted_model" plus validation_warning so the
        # UI can say why instead of silently dropping the symbol.
        screened.sort(key=_screen_sort_key, reverse=True)
        return screened

    def run_backtest(self, symbol, category=None, stock_mode="US", days_back=90, request_context=None):
        self._enforce_public_rate_limit(
            "run_backtest", request_context,
            self.security_config.backtest_limit_per_minute, self.security_config.backtest_limit_per_user_minute, 60,
        )
        validated_mode = SecurityValidator.validate_choice(stock_mode, "stock_mode", {"US", "KR"})
        validated_symbol = SecurityValidator.validate_symbol(symbol, validated_mode, self.security_config)
        validated_days = SecurityValidator.validate_int(days_back, "days_back", 30, 3650)

        resolved_category = category or category_for_symbol(validated_symbol)
        detector = self.get_detector(resolved_category, validated_mode)
        df = detector.fetch_market_data(validated_symbol, outputsize="full")
        if df is None or len(df) < validated_days:
            raise ValueError(f"Insufficient data for backtesting {validated_symbol}")
        df_recent = df.tail(validated_days)
        result = walk_forward_backtest(detector, df_recent)
        result["symbol"] = validated_symbol
        result["category"] = resolved_category
        return result
