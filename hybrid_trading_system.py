#!/usr/bin/env python3

import argparse
import warnings
import joblib
import os
import sys
import glob
import requests
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, classification_report, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
import ta

warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    xgb = None
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    lgb = None
    HAS_LGB = False

try:
    from imblearn.over_sampling import SMOTE, ADASYN
    HAS_IMBALANCE = True
except Exception:
    SMOTE = None
    ADASYN = None
    HAS_IMBALANCE = False

ACTION_LABELS = {0: 'SHORT', 1: 'PASS', 2: 'LONG'}
INVERSE_LABELS = {v: k for k, v in ACTION_LABELS.items()}

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class HybridEnsemble(BaseEstimator, ClassifierMixin):
    
    def __init__(self, models_dict=None, calibrators=None):
        self.models_dict = models_dict if models_dict is not None else {}
        self.calibrators = calibrators
        self.classes_ = np.array([0, 1, 2])
        self.n_features_in_ = None
    
    def fit(self, X, y):
        if hasattr(X, 'shape'):
            self.n_features_in_ = X.shape[1]
        return self
    
    def _predict_proba_raw(self, X):
        proba_list = []
        weights = []
        
        if 'xgb' in self.models_dict and HAS_XGB:
            m = self.models_dict['xgb']
            dtest = xgb.DMatrix(X)
            proba_list.append(m.predict(dtest))
            weights.append(0.35)
        
        if 'lgb' in self.models_dict and HAS_LGB:
            m = self.models_dict['lgb']
            proba_list.append(m.predict(X))
            weights.append(0.35)
        
        if 'rf' in self.models_dict:
            m = self.models_dict['rf']
            proba_list.append(m.predict_proba(X))
            weights.append(0.30)
        
        if len(proba_list) == 0:
            raise RuntimeError('No models available for prediction')
        
        weights = np.array(weights) / sum(weights)
        weighted_proba = sum(w * p for w, p in zip(weights, proba_list))
        return weighted_proba
    
    def predict_proba(self, X):
        proba = self._predict_proba_raw(X)
        
        if self.calibrators is not None:
            calibrated = np.zeros_like(proba)
            for i in range(3):
                if i in self.calibrators:
                    proba_i = np.clip(proba[:, i], 1e-7, 1 - 1e-7)
                    calibrated[:, i] = self.calibrators[i].predict(proba_i)
                else:
                    calibrated[:, i] = proba[:, i]
            
            calibrated = np.clip(calibrated, 0, 1)
            row_sums = calibrated.sum(axis=1, keepdims=True)
            proba = calibrated / np.maximum(row_sums, 1e-10)
        
        return proba
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

class DataProcessor:
    """
    Handles loading, validation, and preprocessing of OHLCV market data.
    Supports multiple timestamp formats and ensures data quality.
    """
    
    # Required columns for OHLCV data
    REQUIRED_OHLCV_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
    
    # Supported timestamp column names (in order of preference)
    TIMESTAMP_COLUMNS = ['timestamp', 'Date', 'time', 'date', 'datetime']
    
    @staticmethod
    def load_and_validate_data(file_path):
        """
        Main entry point for loading and validating market data from CSV.
        
        Args:
            file_path (str): Path to CSV file containing OHLCV data
            
        Returns:
            pd.DataFrame: Validated and processed DataFrame with datetime index
            
        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        try:
            df = DataProcessor._read_csv(file_path)
            df = DataProcessor._parse_and_set_timestamp_index(df)
            df = DataProcessor._normalize_column_names(df)
            DataProcessor._validate_required_columns(df)
            df = DataProcessor._convert_to_numeric(df)
            df = DataProcessor._clean_data(df)
            df = DataProcessor._add_derived_columns(df)
            
            print(f"✅ Processed: {len(df)} rows, {df.index.min()} → {df.index.max()}")
            return df
            
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            raise
    
    @staticmethod
    def _read_csv(file_path):
        """Read CSV file and perform initial validation."""
        df = pd.read_csv(file_path)
        print(f"📂 Loaded {len(df)} rows from {file_path}")
        
        if df.empty:
            raise ValueError("CSV file is empty")
        
        return df
    
    @staticmethod
    def _parse_and_set_timestamp_index(df):
        """
        Detect and parse timestamp column, then set as index.
        Tries multiple timestamp formats and column names.
        """
        timestamp_col = DataProcessor._find_timestamp_column(df)
        
        if timestamp_col is None:
            raise ValueError(f"No timestamp column found. Expected one of: {DataProcessor.TIMESTAMP_COLUMNS}")
        
        print(f"🕐 Using '{timestamp_col}' as timestamp column")
        
        # Parse timestamp with multiple strategies
        df[timestamp_col] = DataProcessor._parse_timestamp(df[timestamp_col])
        
        # Set as index
        df.set_index(timestamp_col, inplace=True)
        
        return df
    
    @staticmethod
    def _find_timestamp_column(df):
        """Find the first matching timestamp column in the DataFrame."""
        # Check case-insensitive matches
        df_columns_lower = {col.lower(): col for col in df.columns}
        
        for ts_col in DataProcessor.TIMESTAMP_COLUMNS:
            if ts_col.lower() in df_columns_lower:
                return df_columns_lower[ts_col.lower()]
        
        return None
    
    @staticmethod
    def _parse_timestamp(series):
        """
        Parse timestamp series with multiple fallback strategies.
        
        Tries in order:
        1. Unix millisecond timestamp
        2. Unix second timestamp  
        3. ISO/string datetime parsing
        """
        # Strategy 1: Try as millisecond timestamp
        try:
            parsed = pd.to_datetime(series, unit='ms', errors='coerce')
            if not parsed.isna().all():
                return parsed
        except:
            pass
        
        # Strategy 2: Try as second timestamp
        try:
            parsed = pd.to_datetime(series, unit='s', errors='coerce')
            if not parsed.isna().all():
                return parsed
        except:
            pass
        
        # Strategy 3: Let pandas infer the format
        try:
            parsed = pd.to_datetime(series, errors='coerce')
            if parsed.isna().all():
                raise ValueError("All timestamp values are invalid")
            return parsed
        except Exception as e:
            raise ValueError(f"Failed to parse timestamps: {e}")
    
    @staticmethod
    def _normalize_column_names(df):
        """Normalize column names to lowercase with underscores."""
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
        return df
    
    @staticmethod
    def _validate_required_columns(df):
        """Check that all required OHLCV columns are present."""
        missing = [col for col in DataProcessor.REQUIRED_OHLCV_COLUMNS 
                   if col not in df.columns]
        
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        print(f"✓ All required columns present: {DataProcessor.REQUIRED_OHLCV_COLUMNS}")
    
    @staticmethod
    def _convert_to_numeric(df):
        """Convert OHLCV columns to numeric types, coercing errors to NaN."""
        numeric_columns = DataProcessor.REQUIRED_OHLCV_COLUMNS.copy()
        
        # Include adj_close if it exists
        if 'adj_close' in df.columns:
            numeric_columns.append('adj_close')
        
        for col in numeric_columns:
            if col in df.columns:
                original_type = df[col].dtype
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Count how many values were coerced to NaN
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    print(f"⚠️  Column '{col}': {nan_count} non-numeric values converted to NaN")
        
        return df
    
    @staticmethod
    def _clean_data(df):
        """
        Clean the data by:
        - Removing rows with NaN in required columns
        - Sorting chronologically
        - Removing duplicate timestamps
        """
        initial_len = len(df)
        
        # Remove rows with NaN in required columns
        df = df.dropna(subset=DataProcessor.REQUIRED_OHLCV_COLUMNS)
        dropped_nan = initial_len - len(df)
        if dropped_nan > 0:
            print(f"🧹 Removed {dropped_nan} rows with NaN values")
        
        # Sort chronologically
        df = df.sort_index()
        
        # Remove duplicate timestamps (keep first occurrence)
        duplicates = df.index.duplicated(keep='first')
        duplicate_count = duplicates.sum()
        if duplicate_count > 0:
            print(f"🧹 Removed {duplicate_count} duplicate timestamps")
            df = df[~duplicates]
        
        return df
    
    @staticmethod
    def _add_derived_columns(df):
        """Add derived columns like adj_close if not present."""
        # For crypto/futures, adj_close is same as close
        if 'adj_close' not in df.columns and 'close' in df.columns:
            df['adj_close'] = df['close'].copy()
            print("✓ Added 'adj_close' column (equal to 'close')")
        
        return df

class UnifiedFeatureEngine:
    
    @staticmethod
    def create_features(df):
        df = df.copy()
        X = pd.DataFrame(index=df.index)
        price_col = 'adj_close' if 'adj_close' in df.columns else 'close'
        
        print(f"Creating unified features using {price_col}...")
        
        for period in [1, 3, 5, 10, 20]:
            X[f'ret_{period}'] = df[price_col].pct_change(period).fillna(0)
        
        X['open_close_ret'] = (df['close'] - df['open']) / df['open'].replace(0, 1)
        X['high_low_range'] = (df['high'] - df['low']) / df[price_col].replace(0, 1)
        X['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df[price_col].replace(0, 1)
        X['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df[price_col].replace(0, 1)
        X['body_size'] = abs(df['close'] - df['open']) / df[price_col].replace(0, 1)
        
        X['doji'] = (X['body_size'] / X['high_low_range'].replace(0, 1) < 0.1).astype(int)
        X['hammer'] = ((df['low'] < df[['open', 'close']].min(axis=1)) & 
                       (X['upper_shadow'] < 0.3 * X['high_low_range'])).astype(int)
        
        ma_periods = [7, 21, 50, 100, 200]
        for period in ma_periods:
            if len(df) >= period:
                sma = df[price_col].rolling(period, min_periods=1).mean()
                ema = df[price_col].ewm(span=period, adjust=False).mean()
                
                X[f'sma_{period}'] = sma
                X[f'ema_{period}'] = ema
                X[f'price_sma_{period}_ratio'] = df[price_col] / sma.replace(0, 1)
                X[f'price_ema_{period}_ratio'] = df[price_col] / ema.replace(0, 1)
                X[f'sma_{period}_slope'] = sma.pct_change(periods=5).fillna(0)
                X[f'ema_{period}_slope'] = ema.pct_change(periods=5).fillna(0)
        
        if len(df) >= 50:
            ma7 = df[price_col].rolling(7, min_periods=1).mean()
            ma21 = df[price_col].rolling(21, min_periods=1).mean()
            ma50 = df[price_col].rolling(50, min_periods=1).mean()
            X['ma_cross_7_21'] = (ma7 - ma21) / df[price_col].replace(0, 1)
            X['ma_cross_21_50'] = (ma21 - ma50) / df[price_col].replace(0, 1)
        
        for rsi_period in [14, 21]:
            delta = df[price_col].diff()
            gain = delta.where(delta > 0, 0).rolling(rsi_period, min_periods=1).mean()
            loss = -delta.where(delta < 0, 0).rolling(rsi_period, min_periods=1).mean()
            rs = gain / loss.replace(0, 1)
            X[f'rsi_{rsi_period}'] = (100 - (100 / (1 + rs))) / 100.0
            X[f'rsi_{rsi_period}'] = X[f'rsi_{rsi_period}'].fillna(0.5)
            X[f'rsi_{rsi_period}_change'] = X[f'rsi_{rsi_period}'].diff().fillna(0)
        
        ema12 = df[price_col].ewm(span=12, adjust=False).mean()
        ema26 = df[price_col].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        X['macd'] = macd / df[price_col].replace(0, 1)
        X['macd_signal'] = signal / df[price_col].replace(0, 1)
        X['macd_histogram'] = (macd - signal) / df[price_col].replace(0, 1)
        X['macd_crossover'] = (macd > signal).astype(int)
        
        for roc_period in [10, 20]:
            X[f'roc_{roc_period}'] = df[price_col].pct_change(periods=roc_period).fillna(0)
        
        low_min = df['low'].rolling(14, min_periods=1).min()
        high_max = df['high'].rolling(14, min_periods=1).max()
        X['stoch_k'] = ((df[price_col] - low_min) / (high_max - low_min).replace(0, 1)).fillna(0.5)
        X['stoch_d'] = X['stoch_k'].rolling(3, min_periods=1).mean().fillna(0.5)
        
        X['williams_r'] = X['stoch_k'] - 1
        
        for window in [5, 10, 20]:
            vol = df[price_col].pct_change().rolling(window, min_periods=1).std().fillna(0)
            X[f'price_volatility_{window}'] = vol
            X[f'high_low_volatility_{window}'] = X['high_low_range'].rolling(window, min_periods=1).std().fillna(0)
        
        X['volatility_ratio'] = (X['price_volatility_5'] / X['price_volatility_20'].replace(0, 1)).fillna(1)
        
        prev_close = df[price_col].shift(1)
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - prev_close).abs(),
            (df['low'] - prev_close).abs()
        ], axis=1).max(axis=1)
        X['atr_14'] = tr.rolling(14, min_periods=1).mean().fillna(0)
        X['atr_21'] = tr.rolling(21, min_periods=1).mean().fillna(0)
        X['atr_normalized'] = X['atr_14'] / df[price_col].replace(0, 1)
        
        for bb_period in [20, 50]:
            if len(df) >= bb_period:
                middle = df[price_col].rolling(bb_period, min_periods=1).mean()
                std = df[price_col].rolling(bb_period, min_periods=1).std().fillna(0)
                upper = middle + 2 * std
                lower = middle - 2 * std
                
                X[f'bb_{bb_period}_width'] = (upper - lower) / middle.replace(0, 1)
                X[f'bb_{bb_period}_position'] = (df[price_col] - lower) / (upper - lower).replace(0, 1)
                X[f'bb_{bb_period}_squeeze'] = (X[f'bb_{bb_period}_width'] < 
                                                X[f'bb_{bb_period}_width'].rolling(20, min_periods=1).mean()).astype(int)
        
        X['adx'] = X['atr_14'] / df[price_col].replace(0, 1)
        
        typical_price = (df['high'] + df['low'] + df[price_col]) / 3
        sma_tp = typical_price.rolling(20, min_periods=1).mean()
        mad = (typical_price - sma_tp).abs().rolling(20, min_periods=1).mean()
        X['cci'] = ((typical_price - sma_tp) / (0.015 * mad.replace(0, 1))).fillna(0) / 200.0
        
        if df['volume'].sum() > 0:
            X['volume_log'] = np.log1p(df['volume'])
            
            vol_sma_20 = df['volume'].rolling(20, min_periods=1).mean()
            X['volume_ratio'] = df['volume'] / vol_sma_20.replace(0, 1)
            
            obv = (np.sign(df[price_col].diff()) * df['volume']).fillna(0).cumsum()
            X['obv'] = obv
            X['obv_change'] = obv.diff().fillna(0) / df['volume'].replace(0, 1)
            
            typical_price = (df['high'] + df['low'] + df[price_col]) / 3
            vwap = (typical_price * df['volume']).rolling(20, min_periods=1).sum() / df['volume'].rolling(20, min_periods=1).sum()
            X['price_vwap_ratio'] = df[price_col] / vwap.replace(0, 1)
        else:
            X['volume_log'] = 0
            X['volume_ratio'] = 1
            X['obv'] = 0
            X['obv_change'] = 0
            X['price_vwap_ratio'] = 1
        
        X['price_position'] = (df[price_col] - df['low']) / (df['high'] - df['low']).replace(0, 1)
        
        try:
            if isinstance(df.index, pd.DatetimeIndex):
                X['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
                X['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
                X['dayofweek_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
                X['dayofweek_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
            else:
                X['hour_sin'] = 0
                X['hour_cos'] = 0
                X['dayofweek_sin'] = 0
                X['dayofweek_cos'] = 0
        except:
            X['hour_sin'] = 0
            X['hour_cos'] = 0
            X['dayofweek_sin'] = 0
            X['dayofweek_cos'] = 0
        
        lag_features = ['ret_1', 'rsi_14', 'macd_histogram', 'volume_ratio', 'atr_14']
        for feature in lag_features:
            if feature in X.columns:
                for lag in [1, 2, 3, 5]:
                    X[f'{feature}_lag_{lag}'] = X[feature].shift(lag).fillna(0)
        
        X = X.replace([np.inf, -np.inf], 0).fillna(0)
        
        print(f"Created {len(X.columns)} unified features")
        return X

class RiskManager:
    
    @staticmethod
    def calculate_levels(entry_price, atr_value, target_profit_pct=0.15, 
                        risk_reward_ratio=0.5, min_stop_pct=0.01):
        # Ensure ATR is reasonable (capped at 10% of price)
        atr_value = min(atr_value, entry_price * 0.10)
        
        # Take profit: at least target_profit_pct or 2x ATR
        tp_amount = max(entry_price * target_profit_pct, atr_value * 2)
        take_profit = entry_price + tp_amount
        
        # Stop loss: use ATR-based or percentage-based, whichever is larger
        # But cap it at reasonable levels (2-5% for most stocks)
        sl_amount = min(
            max(atr_value * 1.5, entry_price * min_stop_pct),
            entry_price * 0.05  # Cap at 5%
        )
        stop_loss = entry_price - sl_amount
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_amount': sl_amount,
            'reward_amount': tp_amount,
            'risk_reward_ratio': tp_amount / sl_amount if sl_amount > 0 else 0
        }
    
    @staticmethod
    def calculate_risk_adjusted_return(df, entry_idx, atr_value, lookforward_periods=10):
        price_col = 'adj_close' if 'adj_close' in df.columns else 'close'
        entry_price = df.iloc[entry_idx][price_col]
        
        risk_levels = RiskManager.calculate_levels(entry_price, atr_value)
        stop_loss = risk_levels['stop_loss']
        take_profit = risk_levels['take_profit']
        
        future_slice = df.iloc[entry_idx+1:entry_idx+lookforward_periods+1]
        
        if len(future_slice) == 0:
            return 0, 0, 0
        
        hit_stop_loss_long = (future_slice['low'] <= stop_loss).any()
        hit_take_profit_long = (future_slice['high'] >= take_profit).any()
        
        hit_stop_loss_short = (future_slice['high'] >= stop_loss + 2 * (entry_price - stop_loss)).any()
        hit_take_profit_short = (future_slice['low'] <= entry_price - (take_profit - entry_price)).any()
        
        max_high = future_slice['high'].max()
        min_low = future_slice['low'].min()
        
        upside_potential = (max_high - entry_price) / entry_price
        downside_risk = (entry_price - min_low) / entry_price
        
        if hit_take_profit_long:
            upside_realized = (take_profit - entry_price) / entry_price
        else:
            upside_realized = upside_potential
        
        if hit_stop_loss_long:
            downside_realized = (entry_price - stop_loss) / entry_price
        else:
            downside_realized = downside_risk
        
        risk_adjusted_return = upside_realized - downside_realized
        
        return upside_potential, downside_risk, risk_adjusted_return

def create_trading_labels_fixed(df, mode='short_term', horizon_bars=3, 
                               lookforward_periods=10, min_hold_periods=3,
                               threshold=0.002, fee=0.0005):
    price_col = 'adj_close' if 'adj_close' in df.columns else 'close'
    labels = pd.Series(INVERSE_LABELS['PASS'], index=df.index)
    
    prev_close = df[price_col].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    atr_14 = tr.rolling(14, min_periods=1).mean().fillna(0)
    
    if mode == 'short_term':
        print(f"Creating SHORT-TERM labels (horizon={horizon_bars} bars, threshold={threshold*100}%)")
        
        future_high = df['high'].shift(-1).rolling(horizon_bars).max()
        future_low = df['low'].shift(-1).rolling(horizon_bars).min()
        max_gain = (future_high - df[price_col]) / df[price_col]
        max_loss = (df[price_col] - future_low) / df[price_col]
        
        labels[max_gain > (threshold + fee)] = INVERSE_LABELS['LONG']
        labels[max_loss > (threshold + fee)] = INVERSE_LABELS['SHORT']
        
    elif mode == 'swing':
        print(f"Creating SWING labels (lookforward={lookforward_periods}, min_hold={min_hold_periods})")
        print("Using risk-adjusted profit potential calculation...")
        
        profit_potentials = []
        
        for i in range(len(df) - lookforward_periods):
            current_price = df.iloc[i][price_col]
            current_atr = atr_14.iloc[i]
            
            future_slice = df.iloc[i+min_hold_periods:i+lookforward_periods+1]
            
            if len(future_slice) > 0:
                upside, downside, risk_adj_return = RiskManager.calculate_risk_adjusted_return(
                    df, i, current_atr, lookforward_periods
                )
                
                profit_potentials.append(upside)
                
                adjusted_threshold = max(threshold * 5, 0.05)
                
                if upside >= adjusted_threshold and risk_adj_return > 0:
                    labels.iloc[i] = INVERSE_LABELS['LONG']
                elif downside >= adjusted_threshold and risk_adj_return < 0:
                    labels.iloc[i] = INVERSE_LABELS['SHORT']
        
        if len(profit_potentials) > 0:
            print(f"Average profit potential: {np.mean(profit_potentials):.2%}")
    
    print(f"Label distribution ({mode}):")
    for label, name in ACTION_LABELS.items():
        count = (labels == label).sum()
        pct = count / len(labels) * 100
        print(f"  {name}: {count} ({pct:.2f}%)")
    
    return labels.astype(int)

class HybridTrainer:
    
    def __init__(self, mode='short_term', cv_splits=5, calibrate=True, random_seed=42):
        self.mode = mode
        self.cv_splits = cv_splits
        self.calibrate = calibrate
        self.random_seed = random_seed
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.training_stats = {}
        
    def load_historical_data(self, data_directory):
        print(f"Loading data from {data_directory}...")
        
        csv_files = glob.glob(os.path.join(data_directory, "*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in {data_directory}")
        
        print(f"Found {len(csv_files)} CSV files")
        dataframes = []
        
        for file_path in csv_files:
            try:
                df = DataProcessor.load_and_validate_data(file_path)
                if len(df) > 0:
                    dataframes.append(df)
            except Exception as e:
                print(f"Skipping {file_path}: {e}")
                continue
        
        if not dataframes:
            raise ValueError("No valid data loaded")
        
        combined = pd.concat(dataframes, ignore_index=False)
        combined = combined.sort_index()
        combined = combined[~combined.index.duplicated(keep='first')]
        
        print(f"Combined: {len(combined)} rows, {combined.index.min()} to {combined.index.max()}")
        return combined
    
    def train(self, data_directory, balance_strategy='smote_balanced'):
        print("=" * 60)
        print(f"FIXED HYBRID TRAINING ({self.mode.upper()})")
        print("=" * 60)
        
        df = self.load_historical_data(data_directory)
        
        if len(df) < 500:
            raise ValueError("Need at least 500 samples for training")
        
        X = UnifiedFeatureEngine.create_features(df)
        
        if self.mode == 'short_term':
            y = create_trading_labels_fixed(df, mode='short_term', horizon_bars=3, threshold=0.002)
        else:
            y = create_trading_labels_fixed(df, mode='swing', lookforward_periods=10, 
                                           min_hold_periods=3, threshold=0.002)
        
        valid_idx = y.dropna().index
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
        df = df.loc[valid_idx]
        
        print(f"\nTotal samples: {len(X)}")
        
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        splits = list(tscv.split(X))
        train_idx, test_idx = splits[-1]
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        if len(splits) > 1:
            _, val_idx = splits[-2]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        else:
            split_point = int(len(X_train) * 0.8)
            X_val = X_train.iloc[split_point:]
            y_val = y_train.iloc[split_point:]
        
        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val)
        X_test_s = self.scaler.transform(X_test)
        self.feature_columns = list(X.columns)
        
        if HAS_IMBALANCE and balance_strategy:
            try:
                print(f"\nApplying {balance_strategy}...")
                max_count = np.bincount(y_train).max()
                
                if 'long_boost' in balance_strategy:
                    strategy = {0: max_count, 1: max_count, 2: int(max_count * 1.2)}
                else:
                    strategy = {0: max_count, 1: max_count, 2: max_count}
                
                if 'adasyn' in balance_strategy and ADASYN:
                    balancer = ADASYN(sampling_strategy=strategy, n_neighbors=7, random_state=self.random_seed)
                else:
                    balancer = SMOTE(sampling_strategy=strategy, k_neighbors=7, random_state=self.random_seed)
                
                X_train_s, y_train = balancer.fit_resample(X_train_s, y_train)
                print(f"After balancing: {len(y_train)} samples")
                print(f"Balanced distribution: SHORT={np.mean(y_train==0)*100:.1f}%, "
                      f"PASS={np.mean(y_train==1)*100:.1f}%, LONG={np.mean(y_train==2)*100:.1f}%")
            except Exception as e:
                print(f"Balancing failed: {e}")
        
        class_counts = np.bincount(y_train)
        n_samples = len(y_train)
        base_weights = {i: n_samples / (len(class_counts) * max(1, class_counts[i])) 
                       for i in range(len(class_counts))}
        
        class_weights = {
            0: base_weights[0] * 1.0,
            1: base_weights[1] * 0.7,
            2: base_weights[2] * 2.0
        }
        print(f"\nClass weights: {class_weights}")
        
        sample_weights = np.array([class_weights[int(c)] for c in y_train])
        
        trained_models = {}
        
        if HAS_XGB:
            print("\n[1/3] Training XGBoost...")
            dtrain = xgb.DMatrix(X_train_s, label=y_train, weight=sample_weights)
            dval = xgb.DMatrix(X_val_s, label=y_val)
            
            params = {
                'objective': 'multi:softprob',
                'num_class': 3,
                'eta': 0.005,
                'max_depth': 4,
                'subsample': 0.6,
                'colsample_bytree': 0.6,
                'lambda': 10.0,
                'alpha': 5.0,
                'gamma': 2.0,
                'min_child_weight': 20,
                'max_delta_step': 2,
                'eval_metric': 'mlogloss',
                'verbosity': 0,
                'seed': self.random_seed
            }
            
            evals_result = {}
            bst = xgb.train(
                params, dtrain,
                num_boost_round=10000,
                early_stopping_rounds=300,
                evals=[(dtrain, 'train'), (dval, 'val')],
                evals_result=evals_result,
                verbose_eval=False
            )
            
            print(f"  Trained at iteration {bst.best_iteration}")
            print(f"  Train loss: {evals_result['train']['mlogloss'][-1]:.4f}, "
                  f"Val loss: {evals_result['val']['mlogloss'][-1]:.4f}")
            
            trained_models['xgb'] = bst
        
        if HAS_LGB:
            print("\n[2/3] Training LightGBM...")
            lgb_train = lgb.Dataset(X_train_s, label=y_train, weight=sample_weights)
            lgb_val = lgb.Dataset(X_val_s, label=y_val, reference=lgb_train)
            
            lgb_params = {
                'objective': 'multiclass',
                'num_class': 3,
                'learning_rate': 0.005,
                'num_leaves': 15,
                'feature_fraction': 0.6,
                'bagging_fraction': 0.6,
                'bagging_freq': 5,
                'lambda_l2': 10.0,
                'lambda_l1': 5.0,
                'min_child_samples': 100,
                'min_split_gain': 0.3,
                'verbose': -1,
                'seed': self.random_seed,
                'metric': 'multi_logloss'
            }
            
            evals_result = {}
            gbm = lgb.train(
                lgb_params, lgb_train,
                num_boost_round=10000,
                valid_sets=[lgb_train, lgb_val],
                valid_names=['train', 'val'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=300, verbose=False),
                    lgb.log_evaluation(period=0),
                    lgb.record_evaluation(evals_result)
                ]
            )
            
            print(f"  Trained at iteration {gbm.best_iteration}")
            print(f"  Train loss: {evals_result['train']['multi_logloss'][-1]:.4f}, "
                  f"Val loss: {evals_result['val']['multi_logloss'][-1]:.4f}")
            
            trained_models['lgb'] = gbm
        
        print("\n[3/3] Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=self.random_seed,
            n_jobs=-1,
            class_weight=class_weights,
            bootstrap=True,
            oob_score=True
        )
        rf.fit(X_train_s, y_train, sample_weight=sample_weights)
        print(f"  OOB Score: {rf.oob_score_:.4f}")
        
        trained_models['rf'] = rf
        
        calibrators = None
        if self.calibrate:
            print("\nCalibrating with isotonic regression...")
            ensemble = HybridEnsemble(trained_models)
            proba_val = ensemble._predict_proba_raw(X_val_s)
            
            calibrators = {}
            for i in range(3):
                iso = IsotonicRegression(out_of_bounds='clip')
                y_val_binary = (y_val == i).astype(int)
                iso.fit(proba_val[:, i], y_val_binary)
                calibrators[i] = iso
        
        self.model = HybridEnsemble(trained_models, calibrators)
        
        proba = self.model.predict_proba(X_test_s)
        y_pred = self.model.predict(X_test_s)
        
        print("\n" + "=" * 60)
        print("TEST SET EVALUATION")
        print("=" * 60)
        print(classification_report(y_test, y_pred, 
                                   target_names=[ACTION_LABELS[i] for i in sorted(ACTION_LABELS.keys())],
                                   digits=4))
        
        cm = confusion_matrix(y_test, y_pred)
        print('Confusion Matrix:')
        print('           SHORT    PASS    LONG')
        for i, row in enumerate(cm):
            print(f'{ACTION_LABELS[i]:>5s}  {row[0]:7d} {row[1]:7d} {row[2]:7d}')
        
        test_logloss = log_loss(y_test, proba)
        print(f'\nLog Loss: {test_logloss:.4f}')
        
        self.training_stats = {
            'mode': self.mode,
            'test_logloss': float(test_logloss),
            'test_samples': len(y_test),
            'train_samples': len(y_train),
            'cv_splits': self.cv_splits,
            'calibrated': self.calibrate,
            'balance_strategy': balance_strategy,
            'trained_at': datetime.utcnow().isoformat() + 'Z',
            'feature_count': len(self.feature_columns),
            'models': list(trained_models.keys())
        }
        
        return test_logloss
    
    def save(self, model_path='hybrid_model_fixed.pkl'):
        if self.model is None:
            raise ValueError("No model trained")
        
        artifacts = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'training_stats': self.training_stats
        }
        
        joblib.dump(artifacts, model_path)
        print(f"\n✅ Model saved to {model_path}")
        print(f"   Features: {len(self.feature_columns)}")
        print(f"   Mode: {self.training_stats['mode']}")
        print(f"   Test LogLoss: {self.training_stats['test_logloss']:.4f}")

class StaticStrategyAnalyzer:
    """
    Traditional technical analysis strategy layer.
    Generates trading signals based on proven static rules.
    """
    
    @staticmethod
    def analyze_rsi_signals(rsi, rsi_period=14):
        """RSI-based signals"""
        signals = []
        confidence = 0.0
        
        if rsi < 30:
            signals.append('RSI_OVERSOLD')
            confidence = (30 - rsi) / 30  # 0.0 to 1.0
            action = 'LONG'
        elif rsi > 70:
            signals.append('RSI_OVERBOUGHT')
            confidence = (rsi - 70) / 30  # 0.0 to 1.0
            action = 'SHORT'
        elif 40 <= rsi <= 60:
            signals.append('RSI_NEUTRAL')
            confidence = 0.3
            action = 'PASS'
        else:
            action = 'PASS'
            confidence = 0.2
            
        return action, signals, confidence
    
    @staticmethod
    def analyze_macd_signals(macd, signal, histogram):
        """MACD-based signals"""
        signals = []
        confidence = 0.0
        action = 'PASS'
        
        # Bullish crossover
        if macd > signal and histogram > 0:
            signals.append('MACD_BULLISH')
            confidence = min(abs(histogram) * 100, 1.0)
            action = 'LONG'
        # Bearish crossover
        elif macd < signal and histogram < 0:
            signals.append('MACD_BEARISH')
            confidence = min(abs(histogram) * 100, 1.0)
            action = 'SHORT'
        # Divergence (histogram weakening)
        elif abs(histogram) < 0.001:
            signals.append('MACD_NEUTRAL')
            confidence = 0.2
            
        return action, signals, confidence
    
    @staticmethod
    def analyze_moving_averages(price, ma_short, ma_medium, ma_long):
        """Moving average-based signals"""
        signals = []
        confidence = 0.0
        action = 'PASS'
        
        # Golden cross (short > medium > long)
        if ma_short > ma_medium > ma_long:
            signals.append('MA_GOLDEN_CROSS')
            # Confidence based on separation
            separation = (ma_short - ma_long) / ma_long
            confidence = min(separation * 10, 1.0)
            action = 'LONG'
        # Death cross (short < medium < long)
        elif ma_short < ma_medium < ma_long:
            signals.append('MA_DEATH_CROSS')
            separation = (ma_long - ma_short) / ma_long
            confidence = min(separation * 10, 1.0)
            action = 'SHORT'
        # Price above all MAs (strong uptrend)
        elif price > ma_short > ma_medium:
            signals.append('MA_STRONG_UPTREND')
            confidence = 0.7
            action = 'LONG'
        # Price below all MAs (strong downtrend)
        elif price < ma_short < ma_medium:
            signals.append('MA_STRONG_DOWNTREND')
            confidence = 0.7
            action = 'SHORT'
            
        return action, signals, confidence
    
    @staticmethod
    def analyze_bollinger_bands(price, bb_upper, bb_lower, bb_middle):
        """Bollinger Bands signals"""
        signals = []
        confidence = 0.0
        action = 'PASS'
        
        bb_width = bb_upper - bb_lower
        if bb_width == 0:
            return action, signals, confidence
        position = (price - bb_lower) / bb_width
        
        if price <= bb_lower:
            signals.append('BB_OVERSOLD')
            confidence = min((bb_lower - price) / bb_lower * 10, 1.0)
            action = 'LONG'
        elif price >= bb_upper:
            signals.append('BB_OVERBOUGHT')
            confidence = min((price - bb_upper) / bb_upper * 10, 1.0)
            action = 'SHORT'
        elif position > 0.7:
            signals.append('BB_UPPER_ZONE')
            confidence = 0.5
            action = 'SHORT'
        elif position < 0.3:
            signals.append('BB_LOWER_ZONE')
            confidence = 0.5
            action = 'LONG'
            
        return action, signals, confidence
    
    @staticmethod
    def analyze_momentum(roc_5, roc_10, roc_20):
        """Rate of Change momentum analysis"""
        signals = []
        confidence = 0.0
        action = 'PASS'
        
        # All positive = strong momentum
        if roc_5 > 0 and roc_10 > 0 and roc_20 > 0:
            signals.append('MOMENTUM_STRONG_UP')
            confidence = min((roc_5 + roc_10 + roc_20) / 3 * 10, 1.0)
            action = 'LONG'
        # All negative = strong bearish
        elif roc_5 < 0 and roc_10 < 0 and roc_20 < 0:
            signals.append('MOMENTUM_STRONG_DOWN')
            confidence = min(abs(roc_5 + roc_10 + roc_20) / 3 * 10, 1.0)
            action = 'SHORT'
        # Accelerating (short term stronger)
        elif roc_5 > roc_10 > roc_20:
            signals.append('MOMENTUM_ACCELERATING')
            confidence = 0.6
            action = 'LONG'
        # Decelerating (short term weaker)
        elif roc_5 < roc_10 < roc_20:
            signals.append('MOMENTUM_DECELERATING')
            confidence = 0.6
            action = 'SHORT'
            
        return action, signals, confidence
    
    @staticmethod
    def analyze_volume(volume, volume_sma_20):
        """Volume analysis"""
        signals = []
        confidence = 0.0
        
        if volume_sma_20 == 0:
            return signals, 0.0
        
        volume_ratio = volume / volume_sma_20
        
        if volume_ratio > 2.0:
            signals.append('VOLUME_SPIKE')
            confidence = min((volume_ratio - 1) / 5, 1.0)
        elif volume_ratio > 1.5:
            signals.append('VOLUME_HIGH')
            confidence = 0.6
        elif volume_ratio < 0.5:
            signals.append('VOLUME_LOW')
            confidence = 0.3
            
        return signals, confidence
    
    @staticmethod
    def get_comprehensive_signals(features_dict):
        """
        Analyze all technical indicators and generate comprehensive signals.
        Returns: (action, confidence, all_signals)
        """
        all_signals = []
        action_votes = {'LONG': 0.0, 'SHORT': 0.0, 'PASS': 0.0}
        
        # Extract features (with defaults)
        rsi = features_dict.get('rsi_14', 50) * 100
        macd = features_dict.get('macd', 0)
        macd_signal = features_dict.get('macd_signal', 0)
        macd_diff = features_dict.get('macd_diff', 0)
        
        price = features_dict.get('close', 0)
        ma_20 = features_dict.get('ma_20', price)
        ma_50 = features_dict.get('ma_50', price)
        ma_200 = features_dict.get('ma_200', price)
        
        bb_upper = features_dict.get('bb_upper_20', price * 1.02)
        bb_lower = features_dict.get('bb_lower_20', price * 0.98)
        bb_middle = features_dict.get('bb_middle_20', price)
        
        roc_5 = features_dict.get('roc_5', 0)
        roc_10 = features_dict.get('roc_10', 0)
        roc_20 = features_dict.get('roc_20', 0)
        
        volume = features_dict.get('volume', 0)
        volume_sma = features_dict.get('volume_sma_20', volume)
        
        # Analyze each indicator
        rsi_action, rsi_signals, rsi_conf = StaticStrategyAnalyzer.analyze_rsi_signals(rsi)
        action_votes[rsi_action] += rsi_conf * 0.25  # 25% weight
        all_signals.extend(rsi_signals)
        
        macd_action, macd_signals, macd_conf = StaticStrategyAnalyzer.analyze_macd_signals(
            macd, macd_signal, macd_diff
        )
        action_votes[macd_action] += macd_conf * 0.25  # 25% weight
        all_signals.extend(macd_signals)
        
        ma_action, ma_signals, ma_conf = StaticStrategyAnalyzer.analyze_moving_averages(
            price, ma_20, ma_50, ma_200
        )
        action_votes[ma_action] += ma_conf * 0.20  # 20% weight
        all_signals.extend(ma_signals)
        
        bb_action, bb_signals, bb_conf = StaticStrategyAnalyzer.analyze_bollinger_bands(
            price, bb_upper, bb_lower, bb_middle
        )
        action_votes[bb_action] += bb_conf * 0.15  # 15% weight
        all_signals.extend(bb_signals)
        
        mom_action, mom_signals, mom_conf = StaticStrategyAnalyzer.analyze_momentum(
            roc_5, roc_10, roc_20
        )
        action_votes[mom_action] += mom_conf * 0.15  # 15% weight
        all_signals.extend(mom_signals)
        
        # Final decision based on weighted votes
        final_action = max(action_votes, key=action_votes.get)
        final_confidence = action_votes[final_action] / sum(action_votes.values()) if sum(action_votes.values()) > 0 else 0
        
        return final_action, final_confidence, all_signals

class HybridDetector:
    
    def __init__(self, api_key, model_path='hybrid_model_fixed.pkl'):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query?"
        self.load_model(model_path)
    
    def load_model(self, model_path):
        try:
            artifacts = joblib.load(resource_path(model_path))
            self.model = artifacts['model']
            self.scaler = artifacts['scaler']
            self.feature_columns = artifacts['feature_columns']
            self.training_stats = artifacts['training_stats']
            
            print(f"✅ Model loaded successfully!")
            print(f"   Mode: {self.training_stats['mode']}")
            print(f"   Features: {len(self.feature_columns)}")
            print(f"   Test LogLoss: {self.training_stats['test_logloss']:.4f}")
            print(f"   Trained: {self.training_stats['trained_at']}")
        except Exception as e:
            raise ValueError(f"Error loading model: {e}")
    
    def fetch_data(self, symbol, function="TIME_SERIES_DAILY", outputsize="compact"):
        try:
            params = {
                'function': function,
                'symbol': symbol,
                'outputsize': outputsize,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if "Error Message" in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            if "Note" in data:
                raise ValueError(f"Rate limit: {data['Note']}")
            
            time_series_data = None
            for key in ["Time Series (Daily)", "Time Series (60min)", "Time Series (5min)"]:
                if key in data:
                    time_series_data = data[key]
                    break
            
            if time_series_data is None:
                raise ValueError("No time series data in response")
            
            df_data = []
            for timestamp, values in time_series_data.items():
                df_data.append({
                    'timestamp': pd.to_datetime(timestamp),
                    'open': float(values.get('1. open', 0)),
                    'high': float(values.get('2. high', 0)),
                    'low': float(values.get('3. low', 0)),
                    'close': float(values.get('4. close', 0)),
                    'volume': int(values.get('5. volume', 0))
                })
            
            df = pd.DataFrame(df_data)
            df['adj_close'] = df['close']
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            
            print(f"Fetched {len(df)} data points for {symbol}")
            return df
            
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None
    
    def analyze(self, symbol):
        print(f"\n{'='*60}")
        print(f"ANALYZING: {symbol}")
        print(f"{'='*60}")
        
        df = self.fetch_data(symbol)
        if df is None or len(df) < 100:
            print("Insufficient data")
            return None
        
        X = UnifiedFeatureEngine.create_features(df)
        
        for feature in self.feature_columns:
            if feature not in X.columns:
                X[feature] = 0
        
        X = X[self.feature_columns].fillna(0).replace([np.inf, -np.inf], 0)
        
        latest = X.iloc[-1:].values
        latest_scaled = self.scaler.transform(latest)
        
        # ML Model Prediction
        proba = self.model.predict_proba(latest_scaled)[0]
        ml_prediction = np.argmax(proba)
        ml_confidence = max(proba)
        
        current_price = df.iloc[-1]['close']
        current_volume = df.iloc[-1]['volume']
        price_change_1d = df.iloc[-1]['close'] / df.iloc[-2]['close'] - 1 if len(df) > 1 else 0
        
        rsi = X.iloc[-1].get('rsi_14', 0.5) * 100
        atr_14 = X.iloc[-1].get('atr_14', 0.02)  # Default to 2% if missing
        atr_value = max(atr_14 * current_price, current_price * 0.02)  # At least 2% of price
        
        # Static Strategy Analysis
        features_dict = {
            'rsi_14': X.iloc[-1].get('rsi_14', 0.5),
            'macd': X.iloc[-1].get('macd', 0),
            'macd_signal': X.iloc[-1].get('macd_signal', 0),
            'macd_diff': X.iloc[-1].get('macd_diff', 0),
            'close': current_price,
            'ma_20': X.iloc[-1].get('ma_20', current_price),
            'ma_50': X.iloc[-1].get('ma_50', current_price),
            'ma_200': X.iloc[-1].get('ma_200', current_price),
            'bb_upper_20': X.iloc[-1].get('bb_upper_20', current_price * 1.02),
            'bb_lower_20': X.iloc[-1].get('bb_lower_20', current_price * 0.98),
            'bb_middle_20': X.iloc[-1].get('bb_middle_20', current_price),
            'roc_5': X.iloc[-1].get('roc_5', 0),
            'roc_10': X.iloc[-1].get('roc_10', 0),
            'roc_20': X.iloc[-1].get('roc_20', 0),
            'volume': current_volume,
            'volume_sma_20': X.iloc[-1].get('volume_sma_20', current_volume)
        }
        
        static_action, static_confidence, static_signals = StaticStrategyAnalyzer.get_comprehensive_signals(features_dict)
        static_prediction = INVERSE_LABELS.get(static_action, 1)
        
        # Hybrid Decision Logic
        # Both agree = high confidence
        if ml_prediction == static_prediction:
            final_prediction = ml_prediction
            confidence_level = 'HIGH'
            combined_confidence = (ml_confidence + static_confidence) / 2
            agreement = 'BOTH_AGREE'
        # ML says trade, Static says pass = medium
        elif static_prediction == 1:  # Static says PASS
            final_prediction = ml_prediction
            confidence_level = 'MEDIUM'
            combined_confidence = ml_confidence * 0.7
            agreement = 'ML_ONLY'
        # Static says trade, ML says pass = medium
        elif ml_prediction == 1:  # ML says PASS
            final_prediction = static_prediction
            confidence_level = 'MEDIUM'
            combined_confidence = static_confidence * 0.7
            agreement = 'STATIC_ONLY'
        # They disagree on direction = conflict, default to PASS
        else:
            final_prediction = 1  # PASS
            confidence_level = 'CONFLICT'
            combined_confidence = 0.0
            agreement = 'CONFLICT'
        
        # Final confidence categorization
        if combined_confidence > 0.75:
            confidence = 'HIGH'
        elif combined_confidence > 0.5:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
        
        risk_levels = RiskManager.calculate_levels(
            current_price, 
            atr_value,
            target_profit_pct=0.15
        )
        
        result = {
            'symbol': symbol,
            'timestamp': df.index[-1],
            'current_price': current_price,
            'current_volume': current_volume,
            'price_change_1d': price_change_1d,
            'rsi': rsi,
            'atr': atr_value,
            'prediction': final_prediction,
            'prediction_label': ACTION_LABELS[final_prediction],
            'ml_prediction': ACTION_LABELS[ml_prediction],
            'static_prediction': static_action,
            'agreement': agreement,
            'static_signals': static_signals,
            'probabilities': {
                'SHORT': proba[0],
                'PASS': proba[1],
                'LONG': proba[2]
            },
            'ml_confidence': ml_confidence,
            'static_confidence': static_confidence,
            'combined_confidence': combined_confidence,
            'confidence': confidence,
            'stop_loss': risk_levels['stop_loss'],
            'take_profit': risk_levels['take_profit'],
            'risk_reward_ratio': risk_levels['risk_reward_ratio']
        }
        
        self._print_result(result)
        return result
    
    def _print_result(self, result):
        symbol = result['symbol']
        timestamp = result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        print(f"\nTimestamp: {timestamp}")
        print(f"Price: ${result['current_price']:.2f}")
        print(f"Change (1D): {result['price_change_1d']:.2%}")
        print(f"RSI: {result['rsi']:.1f}")
        print(f"Volume: {result['current_volume']:,}")
        
        # Hybrid Analysis Display
        print(f"\n🔍 HYBRID ANALYSIS:")
        print(f"   Agreement: {result['agreement']}")
        print(f"   🤖 ML Prediction: {result['ml_prediction']} (confidence: {result['ml_confidence']:.1%})")
        print(f"   📊 Static Strategy: {result['static_prediction']} (confidence: {result['static_confidence']:.1%})")
        
        if result['static_signals']:
            print(f"   📈 Technical Signals: {', '.join(result['static_signals'][:5])}")
        
        print(f"\n📊 FINAL PREDICTION: {result['prediction_label']}")
        print(f"   Combined Confidence: {result['combined_confidence']:.1%}")
        print(f"   Confidence Level: {result['confidence']}")
        print(f"   ML Probabilities:")
        for label, prob in result['probabilities'].items():
            bar = '█' * int(prob * 20)
            print(f"     {label:>5s}: {prob:.1%} {bar}")
        
        if result['prediction_label'] != 'PASS':
            print(f"\n💰 RISK MANAGEMENT:")
            print(f"   Entry: ${result['current_price']:.2f}")
            print(f"   Stop Loss: ${result['stop_loss']:.2f} ({(result['stop_loss']/result['current_price']-1):.2%})")
            print(f"   Take Profit: ${result['take_profit']:.2f} ({(result['take_profit']/result['current_price']-1):.2%})")
            print(f"   Risk/Reward: 1:{result['risk_reward_ratio']:.2f}")
            
            if result['prediction_label'] == 'LONG' and result['confidence'] == 'HIGH':
                print(f"\n🔥 STRONG BUY SIGNAL! (Both ML and Technical Analysis Agree)")
            elif result['prediction_label'] == 'SHORT' and result['confidence'] == 'HIGH':
                print(f"\n⚠️  STRONG SELL SIGNAL! (Both ML and Technical Analysis Agree)")
            elif result['agreement'] == 'CONFLICT':
                print(f"\n⚠️  CAUTION: ML and Technical Analysis Disagree - Trade Skipped")
        
        print(f"\n{'='*60}\n")
    
    def monitor(self, symbols, interval=300, alert_threshold=0.7):
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(',')]
        
        print(f"\n🔍 Monitoring {len(symbols)} symbols...")
        print(f"   Symbols: {', '.join(symbols)}")
        print(f"   Interval: {interval}s")
        print(f"   Alert threshold: {alert_threshold*100}%")
        print("   Press Ctrl+C to stop\n")
        
        try:
            while True:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Scanning...")
                
                opportunities = []
                for symbol in symbols:
                    try:
                        result = self.analyze(symbol)
                        if result and result['prediction_label'] != 'PASS':
                            max_prob = max(result['probabilities'].values())
                            if max_prob >= alert_threshold:
                                opportunities.append(result)
                        time.sleep(2)
                    except Exception as e:
                        print(f"Error with {symbol}: {e}")
                
                if opportunities:
                    print(f"\n🔔 FOUND {len(opportunities)} OPPORTUNITIES:")
                    for opp in sorted(opportunities, 
                                    key=lambda x: max(x['probabilities'].values()), 
                                    reverse=True):
                        print(f"  {opp['symbol']:>6s} {opp['prediction_label']:>5s} "
                              f"({max(opp['probabilities'].values()):.1%}) @ ${opp['current_price']:.2f}")
                else:
                    print("No high-confidence opportunities found")
                
                print(f"\nNext scan in {interval}s...\n")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped")
    
    def backtest(self, symbol, days_back=90):
        print(f"\n{'='*60}")
        print(f"BACKTESTING: {symbol} ({days_back} days)")
        print(f"{'='*60}\n")
        
        df = self.fetch_data(symbol, outputsize="full")
        if df is None or len(df) < days_back:
            print("Insufficient data")
            return None
        
        df = df.tail(days_back).copy()
        
        X = UnifiedFeatureEngine.create_features(df)
        for feature in self.feature_columns:
            if feature not in X.columns:
                X[feature] = 0
        
        X = X[self.feature_columns].fillna(0).replace([np.inf, -np.inf], 0)
        
        trades = []
        position = None
        
        for i in range(50, len(df) - 5):
            try:
                features = X.iloc[i:i+1].values
                features_scaled = self.scaler.transform(features)
                proba = self.model.predict_proba(features_scaled)[0]
                prediction = np.argmax(proba)
                
                current_price = df.iloc[i]['close']
                current_date = df.index[i]
                atr = X.iloc[i].get('atr_14', 0) * current_price
                
                if position is None and prediction != 1 and max(proba) >= 0.7:
                    risk_levels = RiskManager.calculate_levels(current_price, atr)
                    
                    position = {
                        'entry_date': current_date,
                        'entry_price': current_price,
                        'direction': ACTION_LABELS[prediction],
                        'probability': max(proba),
                        'stop_loss': risk_levels['stop_loss'],
                        'take_profit': risk_levels['take_profit']
                    }
                
                elif position is not None:
                    days_held = (current_date - position['entry_date']).days
                    
                    exit_reason = None
                    if position['direction'] == 'LONG':
                        if current_price <= position['stop_loss']:
                            exit_reason = 'Stop Loss'
                        elif current_price >= position['take_profit']:
                            exit_reason = 'Take Profit'
                        elif days_held >= 10:
                            exit_reason = 'Max Time'
                    else:
                        short_stop = position['entry_price'] + (position['entry_price'] - position['stop_loss'])
                        short_take_profit = position['entry_price'] - (position['take_profit'] - position['entry_price'])
                        
                        if current_price >= short_stop:
                            exit_reason = 'Stop Loss'
                        elif current_price <= short_take_profit:
                            exit_reason = 'Take Profit'
                        elif days_held >= 10:
                            exit_reason = 'Max Time'
                    
                    if exit_reason:
                        if position['direction'] == 'LONG':
                            profit_pct = (current_price - position['entry_price']) / position['entry_price']
                        else:
                            profit_pct = (position['entry_price'] - current_price) / position['entry_price']
                        
                        trades.append({
                            'entry_date': position['entry_date'],
                            'exit_date': current_date,
                            'direction': position['direction'],
                            'entry_price': position['entry_price'],
                            'exit_price': current_price,
                            'days_held': days_held,
                            'profit_pct': profit_pct,
                            'probability': position['probability'],
                            'exit_reason': exit_reason
                        })
                        position = None
            
            except Exception as e:
                continue
        
        if not trades:
            print("No trades generated")
            return None
        
        profits = [t['profit_pct'] for t in trades]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p <= 0]
        
        win_rate = len(wins) / len(profits) if profits else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        avg_profit = np.mean(profits)
        total_return = np.prod([1 + p for p in profits]) - 1
        
        long_trades = [t for t in trades if t['direction'] == 'LONG']
        short_trades = [t for t in trades if t['direction'] == 'SHORT']
        
        print(f"{'='*60}")
        print("BACKTEST RESULTS")
        print(f"{'='*60}")
        print(f"Total Trades: {len(trades)}")
        print(f"  LONG: {len(long_trades)}")
        print(f"  SHORT: {len(short_trades)}")
        print(f"\nPerformance:")
        print(f"  Win Rate: {win_rate:.1%}")
        print(f"  Avg Win: {avg_win:.2%}")
        print(f"  Avg Loss: {avg_loss:.2%}")
        print(f"  Avg Profit/Trade: {avg_profit:.2%}")
        print(f"  Total Return: {total_return:.2%}")
        print(f"  Best Trade: {max(profits):.2%}")
        print(f"  Worst Trade: {min(profits):.2%}")
        
        print(f"\nExit Reasons:")
        for reason in ['Stop Loss', 'Take Profit', 'Max Time']:
            count = len([t for t in trades if t['exit_reason'] == reason])
            reason_trades = [t for t in trades if t['exit_reason'] == reason]
            reason_win_rate = len([t for t in reason_trades if t['profit_pct'] > 0]) / len(reason_trades) if reason_trades else 0
            print(f"  {reason:>12s}: {count:3d} trades (Win rate: {reason_win_rate:.1%})")
        
        print(f"{'='*60}\n")
        
        return {
            'trades': trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'total_return': total_return,
            'num_trades': len(trades)
        }

def main():
    parser = argparse.ArgumentParser(description='Fixed Hybrid Advanced Trading System')
    parser.add_argument('--mode', choices=['train', 'analyze', 'monitor', 'backtest'], 
                       required=True, help='Operation mode')
    parser.add_argument('--data', default='./historical_data', help='Training data directory')
    parser.add_argument('--symbol', help='Symbol to analyze/backtest')
    parser.add_argument('--symbols', help='Comma-separated symbols to monitor')
    parser.add_argument('--model', default='hybrid_model_fixed.pkl', help='Model file path')
    parser.add_argument('--api-key', default='WEXRA3OHQYO9I592', help='Alpha Vantage API key')
    parser.add_argument('--trading-mode', choices=['short_term', 'swing'], 
                       default='swing', help='Trading mode')
    parser.add_argument('--cv-splits', type=int, default=5, help='CV splits for training')
    parser.add_argument('--balance', default='smote_long_boost', help='Balance strategy')
    parser.add_argument('--interval', type=int, default=300, help='Monitor interval (seconds)')
    parser.add_argument('--days', type=int, default=90, help='Backtest days')
    
    args = parser.parse_args()
    
    print("\n🚀 HYBRID TRADING SYSTEM (Static Strategy + ML Enhancement)")
    print("   ✅ Technical Analysis + Machine Learning")
    print("   ✅ Dual signal validation")
    print("   ✅ Risk management integrated")
    print("   ✅ Conflict detection & filtering\n")
    
    if args.mode == 'train':
        print(f"Training mode: {args.trading_mode}")
        trainer = HybridTrainer(mode=args.trading_mode, cv_splits=args.cv_splits)
        trainer.train(args.data, balance_strategy=args.balance)
        trainer.save(args.model)
    
    elif args.mode == 'analyze':
        if not args.symbol:
            print("Error: --symbol required for analyze mode")
            return
        detector = HybridDetector(args.api_key, args.model)
        detector.analyze(args.symbol)
    
    elif args.mode == 'monitor':
        if not args.symbols:
            print("Error: --symbols required for monitor mode")
            return
        detector = HybridDetector(args.api_key, args.model)
        detector.monitor(args.symbols, interval=args.interval)
    
    elif args.mode == 'backtest':
        if not args.symbol:
            print("Error: --symbol required for backtest mode")
            return
        detector = HybridDetector(args.api_key, args.model)
        detector.backtest(args.symbol, days_back=args.days)

if __name__ == '__main__':
    main()