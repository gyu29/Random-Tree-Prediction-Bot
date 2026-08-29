"""Technical-analysis feature engineering.

Every indicator here is causal (uses only the current row and earlier rows --
rolling windows, shifts, and cumulative stats), so this module introduces no
look-ahead: the leakage issue fixed elsewhere in this rewrite was in the
train/test split methodology, not in these feature computations.

Causal is necessary but not sufficient: a feature also has to be *comparable*
across the train and test windows. Anything carrying an absolute price or volume
level fails that test even though it looks backward only. AAPL's training rows sit
under a dollar (1980-2006) while its test rows sit above $100 (2012-2026), and a
decision tree can only split at values it saw while fitting -- every test row past
the end of the training range collapses into whichever leaf the largest training
value reached. Measured on the pre-fix feature set, 25.7% of growth_tech's test
rows fell outside the entire training range on level features versus 0.3% on the
scale-free ones (sma_200 alone: 44%).

So the level columns are still computed -- app/detector.py sizes stop-loss and
take-profit from `atr_14`, `macd_crossover` is derived from `macd`/`macd_signal`,
and `volume_ratio` needs `volume_sma_20` -- but they are named in
PRICE_LEVEL_COLUMNS and excluded from the model's feature set by app/trainer.py.
Each has a scale-free counterpart that is a feature: `price_sma_N_ratio` and
`sma_N_slope` for the moving averages, `bb_N_width`/`bb_N_position` for the bands,
`*_pct` (divided by price) for ATR and MACD, `price_vwap_ratio` for VWAP, and a
rolling z-score for the two cumulative volume series.
"""
import warnings

import numpy as np
import pandas as pd
import ta

MA_PERIODS = [5, 10, 20, 50, 100, 200]
BB_PERIODS = [20, 50]

# Window for the rolling z-scores that replace the cumulative volume series
# (on_balance_volume, volume_price_trend). Both are running sums with no natural
# scale -- their level depends on how long the series has been running, so the raw
# value is not comparable between a symbol's train and test windows, or between
# two symbols in the same category.
ROLLING_Z_WINDOW = 60

# Absolute price/volume levels: kept in the frame for the consumers named in the
# module docstring, never used as model features. app/trainer.py folds this list
# into EXCLUDE_FROM_FEATURES so there is one source of truth for it.
PRICE_LEVEL_COLUMNS = (
    [f"sma_{period}" for period in MA_PERIODS]
    + [f"ema_{period}" for period in MA_PERIODS]
    + [f"bb_{period}_{edge}" for period in BB_PERIODS for edge in ("upper", "lower", "middle")]
    + [
        "macd", "macd_signal", "macd_histogram",
        "atr_14", "atr_21",
        "volume_sma_20", "on_balance_volume", "volume_price_trend", "volume_weighted_price",
    ]
)


def _rolling_zscore(series, window=ROLLING_Z_WINDOW):
    """Scale-free standing of `series` against its own recent history. A zero
    rolling std (a perfectly flat window) becomes NaN rather than inf; callers
    already fill NaN/inf before fitting."""
    rolling = series.rolling(window, min_periods=window)
    deviation = rolling.std().replace(0.0, np.nan)
    return (series - rolling.mean()) / deviation


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

        for period in MA_PERIODS:
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
        # MACD is a difference of two price EMAs, so it inherits the price scale.
        df["macd_pct"] = df["macd"] / df[price_col]
        df["macd_signal_pct"] = df["macd_signal"] / df[price_col]
        df["macd_histogram_pct"] = df["macd_histogram"] / df[price_col]

        for period in BB_PERIODS:
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
            df["price_vwap_ratio"] = df[price_col] / df["volume_weighted_price"]
            df["on_balance_volume_z"] = _rolling_zscore(df["on_balance_volume"])
            df["volume_price_trend_z"] = _rolling_zscore(df["volume_price_trend"])
        else:
            df["volume_sma_20"] = 1
            df["volume_ratio"] = 1
            df["volume_price_trend"] = 0
            df["on_balance_volume"] = 0
            df["volume_weighted_price"] = df[price_col]
            df["price_vwap_ratio"] = 1.0
            df["on_balance_volume_z"] = 0.0
            df["volume_price_trend_z"] = 0.0

        df["atr_14"] = ta.volatility.average_true_range(df["high"], df["low"], df[price_col], window=14)
        df["atr_21"] = ta.volatility.average_true_range(df["high"], df["low"], df[price_col], window=21)
        df["atr_14_pct"] = df["atr_14"] / df[price_col]
        df["atr_21_pct"] = df["atr_21"] / df[price_col]
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
        # Lags of the scale-free variants: lagging `atr_14`/`macd_histogram` directly
        # would reintroduce four price-scaled features apiece under a lag_ name.
        lag_features = ["price_change", "rsi_14", "macd_histogram_pct", "volume_ratio", "atr_14_pct"]
        for feature in lag_features:
            if feature in df.columns:
                for lag in lag_periods:
                    df[f"{feature}_lag_{lag}"] = df[feature].shift(lag)

        for window in [5, 10, 20]:
            df[f"price_volatility_{window}"] = df["price_change"].rolling(window).std()
            df[f"price_momentum_{window}"] = df[price_col].pct_change(periods=window)
            df[f"high_low_volatility_{window}"] = df["high_low_range"].rolling(window).std()

        return df
