import math

import numpy as np
import pandas as pd

from app import model_registry
from app.config import DEFAULT_DECISION_THRESHOLD, DEFAULT_LOOKFORWARD_PERIODS, DEFAULT_SWING_THRESHOLD
from app.indicators import TechnicalIndicators
from app.labeling import effective_threshold


class NoModelAvailableError(Exception):
    """No trained model exists yet for this category -- an expected, recoverable state."""


class PredictionError(Exception):
    """A trained model exists but prediction raised. A real bug; never silently replaced."""


def _default_for_missing_feature(feature, df_features):
    if "sma" in feature or "ema" in feature:
        return df_features["close"].rolling(20).mean().fillna(df_features["close"])
    if "rsi" in feature:
        return 50
    if "volume" in feature:
        return 1
    if "price_change" in feature:
        return 0
    if "ratio" in feature:
        return 1
    return 0


class SwingTradeDetector:
    """Holds one category's model + a market data provider for fetching symbols to score."""

    def __init__(self, category, provider):
        self.category = category
        self.provider = provider
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.training_stats = {}
        self.manifest = None
        self.version_warnings = []
        self.swing_threshold = DEFAULT_SWING_THRESHOLD
        self.effective_swing_threshold = effective_threshold(DEFAULT_SWING_THRESHOLD)
        self.lookforward_periods = DEFAULT_LOOKFORWARD_PERIODS
        self.decision_threshold = DEFAULT_DECISION_THRESHOLD
        self._load_error = None
        self._load()

    def _load(self):
        try:
            loaded = model_registry.load(self.category)
        except model_registry.ModelNotFoundError as error:
            self._load_error = str(error)
            return

        self.model = loaded.model
        self.scaler = loaded.scaler
        self.feature_columns = loaded.feature_columns
        self.training_stats = loaded.training_stats
        self.manifest = loaded.manifest
        self.version_warnings = loaded.version_warnings

        self.swing_threshold = self.training_stats.get("swing_threshold", DEFAULT_SWING_THRESHOLD)
        self.effective_swing_threshold = self.training_stats.get(
            "effective_swing_threshold", effective_threshold(self.swing_threshold)
        )
        self.lookforward_periods = self.training_stats.get("lookforward_periods", DEFAULT_LOOKFORWARD_PERIODS)
        self.decision_threshold = self.training_stats.get("decision_threshold", DEFAULT_DECISION_THRESHOLD)
        if hasattr(self.model, "decision_threshold"):
            self.model.decision_threshold = self.decision_threshold

    @property
    def is_ready(self):
        return self.model is not None

    def fetch_market_data(self, symbol, **kwargs):
        return self.provider.fetch_ohlcv(symbol, **kwargs)

    def resolve_company_name(self, symbol):
        return self.provider.resolve_company_name(symbol)

    def calculate_stop_take_profit(self, entry_price, atr_value, risk_reward_ratio=0.5):
        """Stop-loss/take-profit sizing. Uses effective_swing_threshold -- the exact same
        value labels were trained on -- so this always matches what the model was trained
        to detect (see app/labeling.py's module docstring for the mismatch this replaces)."""
        threshold = self.effective_swing_threshold
        tp_amount = max(entry_price * threshold, atr_value * 2)
        take_profit = entry_price + tp_amount
        sl_amount = max(tp_amount * risk_reward_ratio, entry_price * 0.01, atr_value)
        stop_loss = entry_price - sl_amount
        return stop_loss, take_profit

    def _build_feature_row(self, df):
        df_features = TechnicalIndicators.create_all_indicators(df)
        df_features = df_features.ffill().bfill().fillna(0).replace([np.inf, -np.inf], 0)
        for feature in self.feature_columns:
            if feature not in df_features.columns:
                df_features[feature] = _default_for_missing_feature(feature, df_features)
        feature_data = df_features[self.feature_columns].fillna(0).replace([np.inf, -np.inf], 0)
        return df_features, feature_data

    def detect_swing_opportunity(self, df, symbol="Unknown"):
        if not self.is_ready:
            raise NoModelAvailableError(
                f"No trained model available for category '{self.category}': {self._load_error}"
            )
        if len(df) < 100:
            raise ValueError(f"Insufficient data for {symbol} (need at least 100 data points, got {len(df)})")

        try:
            df_features, feature_data = self._build_feature_row(df)
            latest_scaled = self.scaler.transform(feature_data.iloc[-1:])
            prediction = self.model.predict(latest_scaled)[0]
            probabilities = self.model.predict_proba(latest_scaled)[0]
        except Exception as error:
            raise PredictionError(f"Prediction failed for {symbol}: {error}") from error

        current_price = df_features.iloc[-1]["close"]
        atr_value = df_features.iloc[-1].get("atr_14", 0)
        stop_loss, take_profit = self.calculate_stop_take_profit(current_price, atr_value)

        return {
            "symbol": symbol,
            "category": self.category,
            "timestamp": df_features.index[-1],
            "current_price": current_price,
            "current_volume": df_features.iloc[-1]["volume"],
            "price_change_1d": df_features.iloc[-1].get("price_change", 0),
            "rsi": df_features.iloc[-1].get("rsi_14", 50),
            "prediction": prediction,
            "swing_probability": probabilities[1] if len(probabilities) > 1 else 0,
            "no_swing_probability": probabilities[0],
            "is_swing_opportunity": prediction == 1,
            "confidence_level": "High" if max(probabilities) > 0.8 else "Medium" if max(probabilities) > 0.6 else "Low",
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "source": "model",
        }


def _max_drawdown(equity_curve):
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    worst = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        worst = min(worst, (value - peak) / peak if peak else 0.0)
    return worst


def walk_forward_backtest(detector, df, decision_threshold=None, decision_start=50):
    """Runs the model bar-by-bar over df and simulates entries/exits.

    No per-bar try/except around the model call: a prediction failure aborts the
    backtest with the real exception. See module docstring.
    """
    if not detector.is_ready:
        raise NoModelAvailableError(f"No trained model available for category '{detector.category}'")

    decision_threshold = detector.decision_threshold if decision_threshold is None else decision_threshold
    lookforward_periods = detector.lookforward_periods

    features = TechnicalIndicators.create_all_indicators(df)
    features = features.ffill().bfill().replace([np.inf, -np.inf], 0)
    for feature in detector.feature_columns:
        if feature not in features.columns:
            features[feature] = 0

    decision_end = len(features) - lookforward_periods
    min_rows_needed = decision_start + lookforward_periods + 1
    if decision_end <= decision_start:
        raise ValueError(
            f"Only {len(features)} usable rows; need at least {min_rows_needed} to allow a full "
            f"{lookforward_periods}-bar hold. Pick a longer window."
        )

    # Batched once, not per-bar. scaler.transform/predict_proba's actual prediction work
    # scales with row count either way, but the fixed per-call overhead (input validation,
    # RF's n_jobs dispatch, etc.) doesn't -- calling it once per bar in a loop made that
    # overhead dominate over the years-long backtests this runs against.
    decision_window = features.iloc[decision_start:decision_end][detector.feature_columns]
    scaled_window = detector.scaler.transform(decision_window)
    window_probabilities = detector.model.predict_proba(scaled_window)[:, 1]

    trades = []
    position = None
    realized_equity = 1.0
    equity = []
    peak_probability = 0.0

    for index in range(len(features)):
        row = features.iloc[index]
        date = features.index[index]
        price = float(row["close"])

        if decision_start <= index < decision_end:
            probability = float(window_probabilities[index - decision_start])
            peak_probability = max(peak_probability, probability)

            if position is None and probability >= decision_threshold:
                stop, take = detector.calculate_stop_take_profit(price, float(row.get("atr_14", 0)))
                position = {
                    "entry_date": date, "entry_index": index, "entry_price": price,
                    "entry_probability": probability, "stop_loss": stop, "take_profit": take,
                }
            elif position is not None:
                bars_held = index - position["entry_index"]
                days_held = (date - position["entry_date"]).days
                profit = (price - position["entry_price"]) / position["entry_price"]
                reason = None
                if price <= position["stop_loss"]:
                    reason = "Stop-loss"
                elif price >= position["take_profit"]:
                    reason = "Take-profit"
                elif bars_held >= lookforward_periods:
                    reason = "Max time"
                if reason:
                    trades.append({
                        "entry_date": position["entry_date"], "exit_date": date,
                        "entry_price": position["entry_price"], "exit_price": price,
                        "days_held": days_held, "profit_pct": profit,
                        "entry_probability": position["entry_probability"], "exit_reason": reason,
                    })
                    realized_equity *= (1 + profit)
                    position = None

        if position is not None:
            open_profit = (price - position["entry_price"]) / position["entry_price"]
            equity.append(realized_equity * (1 + open_profit))
        else:
            equity.append(realized_equity)

    profits = [trade["profit_pct"] for trade in trades]
    returns = np.asarray(profits or [0.0])
    first_price = float(df.iloc[0]["close"]) if len(df) else 0.0

    no_trades_reason = None
    if not trades:
        no_trades_reason = (
            f"No trades triggered: predicted swing probability peaked at {peak_probability:.0%} in this "
            f"window, never reaching the {decision_threshold:.0%} decision threshold. Try a lower threshold, "
            "a longer window, or a more volatile symbol."
        )

    return {
        "trades": trades,
        "win_rate": len([p for p in profits if p > 0]) / len(profits) if profits else 0,
        "avg_profit": float(np.mean(profits)) if profits else 0.0,
        "total_return": equity[-1] - 1 if equity else 0.0,
        "num_trades": len(trades),
        "sharpe": float(np.mean(returns) / np.std(returns) * math.sqrt(12)) if np.std(returns) else 0.0,
        "max_drawdown": _max_drawdown(equity),
        "equity_curve": equity,
        "buy_hold_curve": [float(price) / first_price for price in df["close"]] if first_price else [],
        "peak_probability": peak_probability,
        "no_trades_reason": no_trades_reason,
    }
