"""Technical-analysis feature engineering.

Every indicator here is causal (uses only the current row and earlier rows --
rolling windows, shifts, and cumulative stats), so this module introduces no
look-ahead: the leakage issue fixed elsewhere in this rewrite was in the
train/test split methodology, not in these feature computations.
"""
import warnings

import pandas as pd
import ta

# Benign: building ~100 indicator columns one assignment at a time fragments the
# DataFrame's internal block layout. Doesn't affect correctness, just a perf hint;
# silenced so it doesn't drown out real training output across dozens of symbols.
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


class TechnicalIndicators:
    @staticmethod
    def create_all_indicators(df):
        df = df.copy()
        price_col = "adj_close" if "adj_close" in df.columns else "close"

        df["price_change"] = df[price_col].pct_change()
        df["high_low_range"] = (df["high"] - df["low"]) / df[price_col]
        df["open_close_change"] = (df[price_col] - df["open"]) / df["open"]
        df["body_size"] = abs(df[price_col] - df["open"]) / df[price_col]
        df["upper_shadow"] = (df["high"] - df[["open", price_col]].max(axis=1)) / df[price_col]
        df["lower_shadow"] = (df[["open", price_col]].min(axis=1) - df["low"]) / df[price_col]

        ma_periods = [5, 10, 20, 50, 100, 200]
        for period in ma_periods:
            if len(df) >= period:
                df[f"sma_{period}"] = ta.trend.sma_indicator(df[price_col], window=period)
                df[f"ema_{period}"] = ta.trend.ema_indicator(df[price_col], window=period)
                df[f"price_sma_{period}_ratio"] = df[price_col] / df[f"sma_{period}"]
                df[f"price_ema_{period}_ratio"] = df[price_col] / df[f"ema_{period}"]
                df[f"sma_{period}_slope"] = df[f"sma_{period}"].pct_change(periods=5)
                df[f"ema_{period}_slope"] = df[f"ema_{period}"].pct_change(periods=5)

        for rsi_period in [14, 21]:
            df[f"rsi_{rsi_period}"] = ta.momentum.rsi(df[price_col], window=rsi_period)
            df[f"rsi_{rsi_period}_change"] = df[f"rsi_{rsi_period}"].diff()

        macd_12_26 = ta.trend.MACD(df[price_col], window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd_12_26.macd()
        df["macd_signal"] = macd_12_26.macd_signal()
        df["macd_histogram"] = macd_12_26.macd_diff()
        df["macd_crossover"] = (df["macd"] > df["macd_signal"]).astype(int)

        bb_periods = [20, 50]
        for period in bb_periods:
            if len(df) >= period:
                bb = ta.volatility.BollingerBands(df[price_col], window=period, window_dev=2)
                df[f"bb_{period}_upper"] = bb.bollinger_hband()
                df[f"bb_{period}_lower"] = bb.bollinger_lband()
                df[f"bb_{period}_middle"] = bb.bollinger_mavg()
                df[f"bb_{period}_width"] = (df[f"bb_{period}_upper"] - df[f"bb_{period}_lower"]) / df[f"bb_{period}_middle"]
                df[f"bb_{period}_position"] = (df[price_col] - df[f"bb_{period}_lower"]) / (
                    df[f"bb_{period}_upper"] - df[f"bb_{period}_lower"]
                )
                df[f"bb_{period}_squeeze"] = (
                    df[f"bb_{period}_width"] < df[f"bb_{period}_width"].rolling(20).mean()
                ).astype(int)

        if df["volume"].sum() > 0 and not df["volume"].isna().all():
            df["volume_sma_20"] = ta.trend.sma_indicator(df["volume"], window=20)
            df["volume_ratio"] = df["volume"] / df["volume_sma_20"]
            df["volume_price_trend"] = ta.volume.volume_price_trend(df[price_col], df["volume"])
            df["on_balance_volume"] = ta.volume.on_balance_volume(df[price_col], df["volume"])
            df["volume_weighted_price"] = ta.volume.volume_weighted_average_price(
                df["high"], df["low"], df[price_col], df["volume"]
            )
        else:
            df["volume_sma_20"] = 1
            df["volume_ratio"] = 1
            df["volume_price_trend"] = 0
            df["on_balance_volume"] = 0
            df["volume_weighted_price"] = df[price_col]

        df["atr_14"] = ta.volatility.average_true_range(df["high"], df["low"], df[price_col], window=14)
        df["atr_21"] = ta.volatility.average_true_range(df["high"], df["low"], df[price_col], window=21)
        df["volatility_ratio"] = df["atr_14"] / df["atr_21"]
        df["roc_10"] = ta.momentum.roc(df[price_col], window=10)
        df["roc_20"] = ta.momentum.roc(df[price_col], window=20)
        df["williams_r"] = ta.momentum.williams_r(df["high"], df["low"], df[price_col], lbp=14)
        df["stoch_k"] = ta.momentum.stoch(df["high"], df["low"], df[price_col])
        df["stoch_d"] = ta.momentum.stoch_signal(df["high"], df["low"], df[price_col])
        df["adx"] = ta.trend.adx(df["high"], df["low"], df[price_col], window=14)
        df["cci"] = ta.trend.cci(df["high"], df["low"], df[price_col], window=20)
        df["doji"] = (abs(df[price_col] - df["open"]) / (df["high"] - df["low"]) < 0.1).astype(int)
        df["hammer"] = (
            (df["low"] < df[["open", price_col]].min(axis=1))
            & (df["high"] - df[["open", price_col]].max(axis=1) < 0.3 * (df["high"] - df["low"]))
        ).astype(int)

        lag_periods = [1, 2, 3, 5]
        lag_features = ["price_change", "rsi_14", "macd_histogram", "volume_ratio", "atr_14"]
        for feature in lag_features:
            if feature in df.columns:
                for lag in lag_periods:
                    df[f"{feature}_lag_{lag}"] = df[feature].shift(lag)

        for window in [5, 10, 20]:
            df[f"price_volatility_{window}"] = df["price_change"].rolling(window).std()
            df[f"price_momentum_{window}"] = df[price_col].pct_change(periods=window)
            df[f"high_low_volatility_{window}"] = df["high_low_range"].rolling(window).std()

        return df
