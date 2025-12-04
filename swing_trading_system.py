import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import ta
import joblib
import os
import glob
import requests
import time
import warnings
from datetime import datetime, timedelta
import sys
warnings.filterwarnings('ignore')

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class DataProcessor:
    @staticmethod
    def load_and_validate_data(file_path):
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded data from {file_path}")
            print(f"Original shape: {df.shape}")
            print(f"Original columns: {list(df.columns)}")
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            elif df.index.name == 'Date' or isinstance(df.index, pd.DatetimeIndex):
                pass
            else:
                try:
                    df.index = pd.to_datetime(df.index)
                except:
                    print("Warning: Could not convert index to datetime")
            df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
            column_mapping = {
                'open': 'open',
                'high': 'high', 
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'dividends': 'dividends',
                'stock_splits': 'stock_splits'
            }
            if 'adj_close' not in df.columns and 'close' in df.columns:
                df['adj_close'] = df['close'].copy()
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            numeric_columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=required_columns)
            df = df.sort_index()
            df = df[~df.index.duplicated(keep='first')]
            print(f"Processed shape: {df.shape}")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            print(f"Sample data:")
            print(df.head())
            return df
        except Exception as e:
            print(f"Error loading data from {file_path}: {str(e)}")
            raise

class TechnicalIndicators:
    @staticmethod
    def create_all_indicators(df):
        df = df.copy()
        price_col = 'adj_close' if 'adj_close' in df.columns else 'close'
        print(f"Creating technical indicators using {price_col} column")
        df['price_change'] = df[price_col].pct_change()
        df['high_low_range'] = (df['high'] - df['low']) / df[price_col]
        df['open_close_change'] = (df[price_col] - df['open']) / df['open']
        df['body_size'] = abs(df[price_col] - df['open']) / df[price_col]
        df['upper_shadow'] = (df['high'] - df[['open', price_col]].max(axis=1)) / df[price_col]
        df['lower_shadow'] = (df[['open', price_col]].min(axis=1) - df['low']) / df[price_col]
        ma_periods = [5, 10, 20, 50, 100, 200]
        for period in ma_periods:
            if len(df) >= period:
                df[f'sma_{period}'] = ta.trend.sma_indicator(df[price_col], window=period)
                df[f'ema_{period}'] = ta.trend.ema_indicator(df[price_col], window=period)
                df[f'price_sma_{period}_ratio'] = df[price_col] / df[f'sma_{period}']
                df[f'price_ema_{period}_ratio'] = df[price_col] / df[f'ema_{period}']
                df[f'sma_{period}_slope'] = df[f'sma_{period}'].pct_change(periods=5)
                df[f'ema_{period}_slope'] = df[f'ema_{period}'].pct_change(periods=5)
        for rsi_period in [14, 21]:
            df[f'rsi_{rsi_period}'] = ta.momentum.rsi(df[price_col], window=rsi_period)
            df[f'rsi_{rsi_period}_change'] = df[f'rsi_{rsi_period}'].diff()
        macd_12_26 = ta.trend.MACD(df[price_col], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd_12_26.macd()
        df['macd_signal'] = macd_12_26.macd_signal()
        df['macd_histogram'] = macd_12_26.macd_diff()
        df['macd_crossover'] = (df['macd'] > df['macd_signal']).astype(int)
        bb_periods = [20, 50]
        for period in bb_periods:
            if len(df) >= period:
                bb = ta.volatility.BollingerBands(df[price_col], window=period, window_dev=2)
                df[f'bb_{period}_upper'] = bb.bollinger_hband()
                df[f'bb_{period}_lower'] = bb.bollinger_lband()
                df[f'bb_{period}_middle'] = bb.bollinger_mavg()
                df[f'bb_{period}_width'] = (df[f'bb_{period}_upper'] - df[f'bb_{period}_lower']) / df[f'bb_{period}_middle']
                df[f'bb_{period}_position'] = (df[price_col] - df[f'bb_{period}_lower']) / (df[f'bb_{period}_upper'] - df[f'bb_{period}_lower'])
                df[f'bb_{period}_squeeze'] = (df[f'bb_{period}_width'] < df[f'bb_{period}_width'].rolling(20).mean()).astype(int)
        if df['volume'].sum() > 0 and not df['volume'].isna().all():
            df['volume_sma_20'] = ta.trend.sma_indicator(df['volume'], window=20)
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            df['volume_price_trend'] = ta.volume.volume_price_trend(df[price_col], df['volume'])
            df['on_balance_volume'] = ta.volume.on_balance_volume(df[price_col], df['volume'])
            df['volume_weighted_price'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df[price_col], df['volume'])
        else:
            df['volume_sma_20'] = 1
            df['volume_ratio'] = 1
            df['volume_price_trend'] = 0
            df['on_balance_volume'] = 0
            df['volume_weighted_price'] = df[price_col]
        df['atr_14'] = ta.volatility.average_true_range(df['high'], df['low'], df[price_col], window=14)
        df['atr_21'] = ta.volatility.average_true_range(df['high'], df['low'], df[price_col], window=21)
        df['volatility_ratio'] = df['atr_14'] / df['atr_21']
        df['roc_10'] = ta.momentum.roc(df[price_col], window=10)
        df['roc_20'] = ta.momentum.roc(df[price_col], window=20)
        df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df[price_col], lbp=14)
        df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df[price_col])
        df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df[price_col])
        df['adx'] = ta.trend.adx(df['high'], df['low'], df[price_col], window=14)
        df['cci'] = ta.trend.cci(df['high'], df['low'], df[price_col], window=20)
        df['doji'] = (abs(df[price_col] - df['open']) / (df['high'] - df['low']) < 0.1).astype(int)
        df['hammer'] = ((df['low'] < df[['open', price_col]].min(axis=1)) & 
                       (df['high'] - df[['open', price_col]].max(axis=1) < 0.3 * (df['high'] - df['low']))).astype(int)
        lag_periods = [1, 2, 3, 5]
        lag_features = ['price_change', 'rsi_14', 'macd_histogram', 'volume_ratio', 'atr_14']
        for feature in lag_features:
            if feature in df.columns:
                for lag in lag_periods:
                    df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        for window in [5, 10, 20]:
            df[f'price_volatility_{window}'] = df['price_change'].rolling(window).std()
            df[f'price_momentum_{window}'] = df[price_col].pct_change(periods=window)
            df[f'high_low_volatility_{window}'] = df['high_low_range'].rolling(window).std()
        print(f"Created {len([col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'adj_close', 'volume', 'dividends', 'stock_splits']])} technical indicators")
        return df

class SwingTradeTrainer:
    def __init__(self, swing_threshold=0.15, lookforward_periods=10, min_hold_periods=3):
        self.swing_threshold = swing_threshold
        self.lookforward_periods = lookforward_periods
        self.min_hold_periods = min_hold_periods
        self.model = RandomForestClassifier(
            n_estimators=100,  # Reduced for faster training with small positive class
            max_depth=8,       # Reduced to prevent overfitting
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight={0: 1, 1: 50},  # Heavy weight for positive class
            bootstrap=True,
            oob_score=True
        )
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        self.training_stats = {}
    def load_historical_data(self, data_directory):
        """Load and process historical data from directory"""
        print(f"Loading data from directory: {data_directory}")
        supported_extensions = ['*.csv', '*.parquet', '*.xlsx']
        all_files = []
        for extension in supported_extensions:
            all_files.extend(glob.glob(os.path.join(data_directory, extension)))
        if not all_files:
            raise ValueError(f"No supported data files found in {data_directory}")
        print(f"Found {len(all_files)} data files")
        dataframes = []
        for file_path in all_files:
            try:
                df = DataProcessor.load_and_validate_data(file_path)
                if len(df) > 0:
                    dataframes.append(df)
                    print(f"Successfully loaded {len(df)} rows from {os.path.basename(file_path)}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        if not dataframes:
            raise ValueError("No valid data files could be loaded")
        combined_df = pd.concat(dataframes, ignore_index=False)
        combined_df = combined_df.sort_index()
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        print(f"Combined dataset: {len(combined_df)} rows")
        print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
        return combined_df
    def create_swing_labels(self, df):
        """Create sophisticated swing trade labels with better balance"""
        df = df.copy()
        price_col = 'adj_close' if 'adj_close' in df.columns else 'close'
        print(f"Creating swing labels with {self.swing_threshold*100}% threshold over {self.lookforward_periods} periods")
        df['swing_label'] = 0
        df['swing_profit_potential'] = 0
        df['swing_risk'] = 0
        for i in range(len(df) - self.lookforward_periods):
            current_price = df.iloc[i][price_col]
            future_slice = df.iloc[i+self.min_hold_periods:i+self.lookforward_periods+1]
            if len(future_slice) > 0:
                max_future_high = future_slice['high'].max()
                min_future_low = future_slice['low'].min()
                upside_potential = (max_future_high - current_price) / current_price
                downside_risk = (current_price - min_future_low) / current_price
                adjusted_threshold = max(0.05, self.swing_threshold * 0.5)  # At least 5% or half the original threshold
                if upside_potential >= adjusted_threshold:
                    df.iloc[i, df.columns.get_loc('swing_label')] = 1
                    df.iloc[i, df.columns.get_loc('swing_profit_potential')] = upside_potential
                    df.iloc[i, df.columns.get_loc('swing_risk')] = downside_risk
        swing_count = df['swing_label'].sum()
        total_count = len(df)
        swing_percentage = (swing_count / total_count) * 100
        print(f"Swing opportunities identified: {swing_count} out of {total_count} samples ({swing_percentage:.2f}%)")
        if swing_count > 0:
            print(f"Average profit potential: {df[df['swing_label']==1]['swing_profit_potential'].mean():.2%}")
            print(f"Average risk: {df[df['swing_label']==1]['swing_risk'].mean():.2%}")
        # If still too imbalanced, create synthetic positive samples
        if swing_percentage < 5.0:
            print("Class imbalance detected. Creating additional positive samples...")
            self._balance_labels(df, price_col)
        return df
    def _balance_labels(self, df, price_col):
        """Create more balanced labels by lowering criteria"""
        additional_positives = 0
        target_positive_ratio = 0.1  # Aim for 10% positive samples
        current_positives = df['swing_label'].sum()
        target_positives = int(len(df) * target_positive_ratio)
        if current_positives < target_positives:
            needed_positives = target_positives - current_positives
            for i in range(len(df) - self.lookforward_periods):
                if df.iloc[i]['swing_label'] == 0 and additional_positives < needed_positives:
                    current_price = df.iloc[i][price_col]
                    future_slice = df.iloc[i+self.min_hold_periods:i+self.lookforward_periods+1]
                    if len(future_slice) > 0:
                        max_future_high = future_slice['high'].max()
                        upside_potential = (max_future_high - current_price) / current_price
                        if upside_potential >= 0.03:  # 3% minimum
                            df.iloc[i, df.columns.get_loc('swing_label')] = 1
                            df.iloc[i, df.columns.get_loc('swing_profit_potential')] = upside_potential
                            additional_positives += 1
            print(f"Added {additional_positives} additional positive samples for balance")
    def prepare_training_data(self, df):
        print("Creating technical indicators...")
        df_features = TechnicalIndicators.create_all_indicators(df)
        print("Creating swing labels...")
        df_labeled = self.create_swing_labels(df_features)

        exclude_cols = ['open', 'high', 'low', 'close', 'adj_close', 'volume', 'dividends', 
                       'stock_splits', 'swing_label', 'swing_profit_potential', 'swing_risk']
        feature_cols = [col for col in df_labeled.columns if col not in exclude_cols]
        df_labeled[feature_cols] = df_labeled[feature_cols].fillna(method='ffill').fillna(method='bfill')
        df_labeled[feature_cols] = df_labeled[feature_cols].replace([np.inf, -np.inf], 0)
        df_clean = df_labeled.dropna(subset=feature_cols + ['swing_label'])
        print(f"Feature engineering complete. Features: {len(feature_cols)}")
        X = df_clean[feature_cols]
        y = df_clean['swing_label']
        self.feature_columns = feature_cols
        return X, y, df_clean
    def train(self, data_directory):
        print(f"=== Training Swing Trade Predictor ===")
        print(f"Swing threshold: {self.swing_threshold*100}%")
        print(f"Lookforward periods: {self.lookforward_periods}")
        print(f"Data directory: {data_directory}")
        df = self.load_historical_data(data_directory)
        if len(df) < 500:
            raise ValueError("Insufficient data for training (minimum 500 samples required)")
        X, y, df_processed = self.prepare_training_data(df)
        print(f"Training dataset shape: {X.shape}")
        print(f"Swing opportunities: {sum(y)} out of {len(y)} samples ({sum(y)/len(y)*100:.2f}%)")
        if sum(y) < 10:
            print("⚠️  WARNING: Very few positive samples detected!")
            print("This may lead to poor model performance. Consider:")
            print("- Lowering the swing threshold")
            print("- Using more historical data")
            print("- Adjusting the lookforward period")
        try:
            if sum(y) > 1:  # Need at least 2 positive samples for stratification
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            else:
                # Fallback to regular split if stratification fails
                print("Using regular train-test split due to insufficient positive samples")
                split_index = int(len(X) * 0.8)
                X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
                y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        except ValueError:
            # Fallback for any stratification issues
            print("Stratification failed, using chronological split")
            split_index = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
            y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        print("Training Random Forest model...")
        self.model.fit(X_train_scaled, y_train)
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        oob_score = self.model.oob_score_
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        print(f"\n=== Model Performance ===")
        print(f"Training accuracy: {train_score:.4f}")
        print(f"Test accuracy: {test_score:.4f}")
        print(f"Out-of-bag score: {oob_score:.4f}")
        print(f"\n=== Classification Report ===")
        print(classification_report(y_test, y_pred))
        print(f"\n=== Confusion Matrix ===")
        print(confusion_matrix(y_test, y_pred))
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(f"\n=== Top 20 Most Important Features ===")
        print(feature_importance.head(20))
        self.training_stats = {
            'train_score': train_score,
            'test_score': test_score,
            'oob_score': oob_score,
            'feature_importance': feature_importance,
            'swing_threshold': self.swing_threshold,
            'lookforward_periods': self.lookforward_periods,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        self.is_trained = True
        return test_score
    def save_model(self, model_filename="swing_model_enhanced.pkl", 
                   scaler_filename="swing_scaler_enhanced.pkl",
                   features_filename="feature_columns_enhanced.pkl"):
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        joblib.dump(self.model, model_filename)
        joblib.dump(self.scaler, scaler_filename)
        joblib.dump(self.feature_columns, features_filename)
        joblib.dump(self.training_stats, "training_stats.pkl")
        print(f"Model saved as {model_filename}")
        print(f"Scaler saved as {scaler_filename}")
        print(f"Features saved as {features_filename}")
        print(f"Training stats saved as training_stats.pkl")

class SwingTradeDetector:
    def __init__(self, api_key, model_path="swing_model_enhanced.pkl", 
                 scaler_path="swing_scaler_enhanced.pkl", 
                 features_path="feature_columns_enhanced.pkl"):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query?"
        self.load_model(model_path, scaler_path, features_path)
    def load_model(self, model_path, scaler_path, features_path):
        try:
            self.model = joblib.load(resource_path(model_path))
            self.scaler = joblib.load(resource_path(scaler_path))
            self.feature_columns = joblib.load(resource_path(features_path))
            try:
                self.training_stats = joblib.load(resource_path("training_stats.pkl"))
                print(f"Model loaded successfully!")
                print(f"Training accuracy: {self.training_stats.get('test_score', 'N/A'):.4f}")
                print(f"Features: {len(self.feature_columns)}")
                self.swing_threshold = self.training_stats.get('swing_threshold', 0.15)
                self.lookforward_periods = self.training_stats.get('lookforward_periods', 10)
            except:
                print("Model loaded successfully! (No training stats available)")
                self.swing_threshold = 0.15
                self.lookforward_periods = 10
        except Exception as e:
            raise ValueError(f"Error loading model components: {e}")
    def fetch_alpha_vantage_data(self, symbol, function="TIME_SERIES_DAILY", outputsize="compact", interval="monthly"):
        try:
            commodity_map = {
                'COPPER': 'COPPER',
                'ALUMINUM': 'ALUMINUM',
                'AL': 'ALUMINUM',
                'GOLD': 'XAU',
                'XAU': 'XAU',
                'SILVER': 'XAG',
                'XAG': 'XAG',
            }
            symbol_upper = symbol.upper()
            is_commodity = symbol_upper in commodity_map
            if is_commodity:
                commodity_func = commodity_map[symbol_upper]
                params = {
                    'function': commodity_func,
                    'interval': interval,
                    'apikey': self.api_key
                }
                response = requests.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                if "Error Message" in data:
                    raise ValueError(f"Alpha Vantage API Error: {data['Error Message']}")
                if "Note" in data:
                    raise ValueError(f"Alpha Vantage API Rate Limit: {data['Note']}")
                if "Information" in data:
                    raise ValueError(f"Alpha Vantage API Info: {data['Information']}")
                if 'data' not in data:
                    raise ValueError("No commodity data found in API response")
                df_data = []
                for row in data['data']:
                    try:
                        df_data.append({
                            'timestamp': pd.to_datetime(row['date']),
                            'open': float(row.get('value', 0)),
                            'high': float(row.get('value', 0)),
                            'low': float(row.get('value', 0)),
                            'close': float(row.get('value', 0)),
                            'volume': 0
                        })
                    except Exception as e:
                        continue
                df = pd.DataFrame(df_data)
                if df.empty:
                    print(f"No data returned for commodity {symbol}")
                    return None
                df['adj_close'] = df['close']
                df.set_index('timestamp', inplace=True)
                df = df.sort_index()
                print(f"Fetched {len(df)} data points for {symbol} (commodity)")
                return df
            else:
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
                    raise ValueError(f"Alpha Vantage API Error: {data['Error Message']}")
                if "Note" in data:
                    raise ValueError(f"Alpha Vantage API Rate Limit: {data['Note']}")
                if "Information" in data:
                    raise ValueError(f"Alpha Vantage API Info: {data['Information']}")

                time_series_keys = [
                    "Time Series (Daily)",
                    "Time Series (60min)",
                    "Time Series (5min)",
                    "Weekly Adjusted Time Series",
                    "Monthly Adjusted Time Series"
                ]
                time_series_data = None
                for key in time_series_keys:
                    if key in data:
                        time_series_data = data[key]
                        break
                if time_series_data is None:
                    raise ValueError("No time series data found in API response")
                df_data = []
                for timestamp, values in time_series_data.items():
                    try:
                        df_data.append({
                            'timestamp': pd.to_datetime(timestamp),
                            'open': float(values['1. open']),
                            'high': float(values['2. high']),
                            'low': float(values['3. low']),
                            'close': float(values['4. close']),
                            'volume': int(values['5. volume'])
                        })
                    except KeyError as e:
                        df_data.append({
                            'timestamp': pd.to_datetime(timestamp),
                            'open': float(values.get('1. open', values.get('open', 0))),
                            'high': float(values.get('2. high', values.get('high', 0))),
                            'low': float(values.get('3. low', values.get('low', 0))),
                            'close': float(values.get('4. close', values.get('close', 0))),
                            'volume': int(values.get('5. volume', values.get('volume', 0)))
                        })
                df = pd.DataFrame(df_data)
                df['adj_close'] = df['close']
                df.set_index('timestamp', inplace=True)
                df = df.sort_index()
                print(f"Fetched {len(df)} data points for {symbol}")
                return df
        except requests.exceptions.RequestException as e:
            print(f"Network error fetching data for {symbol}: {e}")
            return None
        except Exception as e:
            print(f"Error fetching Alpha Vantage data for {symbol}: {e}")
            return None

    def _calculate_stop_take_profit(self, entry_price, atr_value, swing_threshold, risk_reward_ratio=0.5):
        """Calculate stop-loss and take-profit levels."""
        tp_amount = max(entry_price * swing_threshold, atr_value * 2)
        take_profit = entry_price + tp_amount

        sl_amount = tp_amount * risk_reward_ratio
        min_sl_amount = max(entry_price * 0.01, atr_value)
        sl_amount = max(sl_amount, min_sl_amount)
        stop_loss = entry_price - sl_amount

        return stop_loss, take_profit

    def detect_swing_opportunity(self, df, symbol="Unknown"):
        try:
            if len(df) < 100:
                print(f"Insufficient data for {symbol} (need at least 100 data points)")
                return None
            df_features = TechnicalIndicators.create_all_indicators(df)
            df_features = df_features.fillna(method='ffill').fillna(method='bfill').fillna(0)
            df_features = df_features.replace([np.inf, -np.inf], 0)
            print(f"Features created for {symbol}: {len(df_features.columns)} columns")
            print(f"Required features: {len(self.feature_columns)}")
            missing_features = []
            for feature in self.feature_columns:
                if feature not in df_features.columns:
                    missing_features.append(feature)
                    if 'sma' in feature or 'ema' in feature:
                        df_features[feature] = df_features['close'].rolling(20).mean().fillna(df_features['close'])
                    elif 'rsi' in feature:
                        df_features[feature] = 50  # Neutral RSI
                    elif 'volume' in feature:
                        df_features[feature] = 1   # Neutral volume ratio
                    elif 'price_change' in feature:
                        df_features[feature] = 0   # No change
                    elif 'ratio' in feature:
                        df_features[feature] = 1   # Neutral ratio
                    else:
                        df_features[feature] = 0   # Default to zero
            if missing_features:
                print(f"Added {len(missing_features)} missing features with defaults for {symbol}")
            df_clean = df_features.copy()
            feature_data = df_clean[self.feature_columns]
            nan_count = feature_data.isna().sum().sum()
            inf_count = np.isinf(feature_data.values).sum()
            if nan_count > 0 or inf_count > 0:
                print(f"Cleaning remaining NaN ({nan_count}) and inf ({inf_count}) values")
                feature_data = feature_data.fillna(0).replace([np.inf, -np.inf], 0)
                df_clean[self.feature_columns] = feature_data
            if len(df_clean) == 0:
                print(f"No data remaining after cleaning for {symbol}")
                return None
            latest_data = df_clean.iloc[-1:][self.feature_columns]
            print(f"Latest data shape: {latest_data.shape}")
            print(f"Data sample: {latest_data.iloc[0][:5].values}")
            try:
                latest_scaled = self.scaler.transform(latest_data)
            except Exception as scale_error:
                print(f"Scaling error for {symbol}: {scale_error}")
                return None
            try:
                prediction = self.model.predict(latest_scaled)[0]
                probabilities = self.model.predict_proba(latest_scaled)[0]
            except Exception as pred_error:
                print(f"Prediction error for {symbol}: {pred_error}")
                return None
            current_price = df_clean.iloc[-1]['close']
            current_volume = df_clean.iloc[-1]['volume']
            timestamp = df_clean.index[-1]
            atr_value = df_clean.iloc[-1].get('atr_14', 0)
            price_change_1d = df_clean.iloc[-1].get('price_change', 0)
            rsi = df_clean.iloc[-1].get('rsi_14', 50)
            stop_loss, take_profit = self._calculate_stop_take_profit(
                current_price, atr_value, self.swing_threshold
            )

            result = {
                'symbol': symbol,
                'timestamp': timestamp,
                'current_price': current_price,
                'current_volume': current_volume,
                'price_change_1d': price_change_1d,
                'rsi': rsi,
                'prediction': prediction,
                'swing_probability': probabilities[1] if len(probabilities) > 1 else 0,
                'no_swing_probability': probabilities[0],
                'is_swing_opportunity': prediction == 1,
                'confidence_level': 'High' if max(probabilities) > 0.8 else 'Medium' if max(probabilities) > 0.6 else 'Low',
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
            return result
        except Exception as e:
            print(f"Error detecting swing opportunity for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None
    def analyze_single_symbol(self, symbol):
        """Analyze a single symbol for swing trade opportunities"""
        print(f"Analyzing {symbol}...")
        df = self.fetch_alpha_vantage_data(symbol, function="TIME_SERIES_DAILY")
        if df is None or len(df) < 50:
            print(f"Insufficient data available for {symbol}")
            return None
        result = self.detect_swing_opportunity(df, symbol)
        if result:
            self.print_analysis_result(result)
        return result
    def print_analysis_result(self, result):
        """Print formatted analysis result including stop-loss and take-profit"""
        symbol = result['symbol']
        timestamp = result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n{'='*60}")
        print(f"SWING TRADE ANALYSIS: {symbol}")
        print(f"{'='*60}")
        print(f"Analysis Time: {timestamp}")
        print(f"Current Price: ${result['current_price']:.2f}")
        print(f"Price Change (1D): {result['price_change_1d']:.2%}")
        print(f"RSI: {result['rsi']:.1f}")
        print(f"Volume: {result['current_volume']:,}")
        print(f"")
        if result['is_swing_opportunity']:
            print(f"🚨 SWING TRADE OPPORTUNITY DETECTED! 🚨")
            print(f"Swing Probability: {result['swing_probability']:.1%}")
            print(f"Confidence Level: {result['confidence_level']}")
            if result['swing_probability'] >= 0.8:
                print(f"🔥 STRONG BUY SIGNAL")
            elif result['swing_probability'] >= 0.6:
                print(f"📈 MODERATE BUY SIGNAL")
            else:
                print(f"⚠️   WEAK BUY SIGNAL")
            print(f"Recommended Stop-Loss: ${result['stop_loss']:.2f}")
            print(f"Recommended Take-Profit: ${result['take_profit']:.2f}")
        else:
            print(f"❌ No swing opportunity detected")
            print(f"No-Swing Probability: {result['no_swing_probability']:.1%}")
            print(f"Confidence Level: {result['confidence_level']}")
        print(f"{'='*60}\n")
    def monitor_multiple_symbols(self, symbols, check_interval=300, alert_threshold=0.7):
        """Monitor multiple symbols for swing trade opportunities"""
        if isinstance(symbols, str):
            symbols = [symbols]
        print(f"Monitoring {len(symbols)} symbols for swing trade opportunities...")
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Check interval: {check_interval} seconds")
        print(f"Alert threshold: {alert_threshold*100}%")
        print("Press Ctrl+C to stop monitoring\n")
        try:
            while True:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{current_time}] Scanning symbols...")
                opportunities = []
                for symbol in symbols:
                    try:
                        result = self.analyze_single_symbol(symbol)
                        if result and result['is_swing_opportunity'] and result['swing_probability'] >= alert_threshold:
                            opportunities.append(result)
                        time.sleep(2)
                    except Exception as e:
                        print(f"Error analyzing {symbol}: {e}")
                        continue
                if opportunities:
                    print(f"\n🔔 FOUND {len(opportunities)} HIGH-PROBABILITY OPPORTUNITIES:")
                    for opp in sorted(opportunities, key=lambda x: x['swing_probability'], reverse=True):
                        print(f"  {opp['symbol']}: {opp['swing_probability']:.1%} at ${opp['current_price']:.2f}")
                        print(f"     SL: ${opp['stop_loss']:.2f}, TP: ${opp['take_profit']:.2f}")
                else:
                    print("No high-probability opportunities found in current scan.")
                print(f"Next scan in {check_interval} seconds...\n")
                time.sleep(check_interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
        except Exception as e:
            print(f"Error during monitoring: {e}")
    def backtest_strategy(self, symbol, days_back=90):
        """Simple backtest of the swing trading strategy with stop-loss and take-profit"""
        print(f"Backtesting swing strategy for {symbol} over {days_back} days...")
        df = self.fetch_alpha_vantage_data(symbol, function="TIME_SERIES_DAILY", outputsize="full")
        if df is None or len(df) < days_back:
            print(f"Insufficient data for backtesting {symbol}")
            return None
        df_recent = df.tail(days_back).copy()
        df_features = TechnicalIndicators.create_all_indicators(df_recent)
        df_features = df_features.fillna(method='ffill').fillna(method='bfill')
        df_features = df_features.replace([np.inf, -np.inf], 0)
        for feature in self.feature_columns:
            if feature not in df_features.columns:
                df_features[feature] = 0

        swing_threshold = getattr(self, 'swing_threshold', 0.15)
        lookforward_periods = getattr(self, 'lookforward_periods', 10)

        trades = []
        position = None
        for i in range(50, len(df_features) - 10):
            try:
                current_data = df_features.iloc[i:i+1][self.feature_columns]
                current_scaled = self.scaler.transform(current_data)
                prediction = self.model.predict(current_scaled)[0]
                probability = self.model.predict_proba(current_scaled)[0][1]
                current_price = df_features.iloc[i]['close']
                current_date = df_features.index[i]
                atr_value = df_features.iloc[i].get('atr_14', 0) 
                if position is None and prediction == 1 and probability >= 0.7:
                    stop_loss, take_profit = self._calculate_stop_take_profit(
                        current_price, atr_value, swing_threshold
                    )
                    position = {
                        'entry_date': current_date,
                        'entry_price': current_price,
                        'entry_probability': probability,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    }
                elif position is not None:
                    days_held = (current_date - position['entry_date']).days
                    profit_pct = (current_price - position['entry_price']) / position['entry_price']
                    exit_reason = None
                    if current_price <= position['stop_loss']:
                        exit_reason = "Stop-Loss"
                    elif current_price >= position['take_profit']:
                        exit_reason = "Take-Profit"
                    elif days_held >= 10:
                        exit_reason = "Max Time"

                    if exit_reason:
                        trade = {
                            'entry_date': position['entry_date'],
                            'exit_date': current_date,
                            'entry_price': position['entry_price'],
                            'exit_price': current_price,
                            'days_held': days_held,
                            'profit_pct': profit_pct,
                            'entry_probability': position['entry_probability'],
                            'exit_reason': exit_reason
                        }
                        trades.append(trade)
                        position = None
            except Exception as e:
                continue
        if not trades:
            print("No trades generated during backtest period")
            return None
        profits = [trade['profit_pct'] for trade in trades]
        win_rate = len([p for p in profits if p > 0]) / len(profits) if profits else 0
        avg_profit = np.mean(profits) if profits else 0
        total_return = np.prod([1 + p for p in profits]) - 1 if profits else 0
        num_trades = len(trades)
        sl_exits = [t for t in trades if t['exit_reason'] == 'Stop-Loss']
        tp_exits = [t for t in trades if t['exit_reason'] == 'Take-Profit']
        time_exits = [t for t in trades if t['exit_reason'] == 'Max Time']
        sl_win_rate = len([t for t in sl_exits if t['profit_pct'] > 0]) / len(sl_exits) if sl_exits else 0
        tp_win_rate = len([t for t in tp_exits if t['profit_pct'] > 0]) / len(tp_exits) if tp_exits else 0
        time_win_rate = len([t for t in time_exits if t['profit_pct'] > 0]) / len(time_exits) if time_exits else 0


        print(f"\n{'='*50}")
        print(f"BACKTEST RESULTS: {symbol}")
        print(f"{'='*50}")
        print(f"Total Trades: {num_trades}")
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Average Profit per Trade: {avg_profit:.2%}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Best Trade: {max(profits):.2%}")
        print(f"Worst Trade: {min(profits):.2%}")
        print("--- Exit Reason Stats ---")
        print(f"Stop-Loss Exits: {len(sl_exits)} (Win Rate: {sl_win_rate:.1%})")
        print(f"Take-Profit Exits: {len(tp_exits)} (Win Rate: {tp_win_rate:.1%})")
        print(f"Max Time Exits: {len(time_exits)} (Win Rate: {time_win_rate:.1%})")

        return {
            'trades': trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'total_return': total_return,
            'num_trades': num_trades
        }

class SwingTradingSystem:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.trainer = None
        self.detector = None
    def train_model(self, data_directory="./historical_data", swing_threshold=0.15, lookforward_periods=10):
        print("Initializing model training...")
        self.trainer = SwingTradeTrainer(
            swing_threshold=swing_threshold,
            lookforward_periods=lookforward_periods
        )
        try:
            test_score = self.trainer.train(data_directory)
            self.trainer.save_model()
            print(f"\n✅ Model training completed successfully!")
            print(f"Final test score: {test_score:.4f}")
            return test_score
        except Exception as e:
            print(f"❌ Error during training: {e}")
            raise
    def initialize_detector(self, model_path="swing_model_enhanced.pkl"):
        if self.api_key is None:
            raise ValueError("API key required for real-time detection")
        try:
            self.detector = SwingTradeDetector(api_key=self.api_key, model_path=model_path)
            print("✅ Swing trade detector initialized successfully!")
            return True
        except Exception as e:
            print(f"❌ Error initializing detector: {e}")
            raise
    def analyze_symbol(self, symbol):
        if self.detector is None:
            self.initialize_detector()
        return self.detector.analyze_single_symbol(symbol)
    def monitor_symbols(self, symbols, check_interval=300, alert_threshold=0.7):
        if self.detector is None:
            self.initialize_detector()
        self.detector.monitor_multiple_symbols(symbols, check_interval, alert_threshold)
    def run_backtest(self, symbol, days_back=90):
        if self.detector is None:
            self.initialize_detector()
        return self.detector.backtest_strategy(symbol, days_back)

if __name__ == "__main__":
    ALPHA_VANTAGE_API_KEY = "WEXRA3OHQYO9I592"
    DATA_DIRECTORY = "./historical_data"
    system = SwingTradingSystem(api_key=ALPHA_VANTAGE_API_KEY)
    print("🚀 Advanced Swing Trading ML System")
    print("=" * 50)
    while True:
        print("\nSelect an option:")
        print("1. Train new model")
        print("2. Analyze single symbol")
        print("3. Monitor multiple symbols")
        print("4. Run backtest")
        print("5. Exit")
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            if choice == "1":
                print("\n🧠 Training new model...")
                try:
                    threshold = float(input("Enter swing threshold (default 0.15): ") or "0.15")
                    periods = int(input("Enter lookforward periods (default 10): ") or "10")
                    system.train_model(
                        data_directory=DATA_DIRECTORY,
                        swing_threshold=threshold,
                        lookforward_periods=periods
                    )
                except Exception as e:
                    print(f"Training failed: {e}")
            elif choice == "2":
                print("\n🧠 Single Symbol Analysis")
                symbol = input("Enter symbol (e.g., AAPL): ").strip().upper()
                if symbol:
                    try:
                        result = system.analyze_symbol(symbol)
                        if not result:
                            print(f"Could not analyze {symbol}")
                    except Exception as e:
                        print(f"Analysis failed: {e}")
            elif choice == "3":
                print("\n🧠 Multi-Symbol Monitoring")
                symbols_input = input("Enter symbols separated by commas (e.g., AAPL,MSFT,GOOGL): ").strip()
                if symbols_input:
                    symbols = [s.strip().upper() for s in symbols_input.split(",")]
                    try:
                        interval = int(input("Check interval in seconds (default 300): ") or "300")
                        threshold = float(input("Alert threshold (default 0.7): ") or "0.7")
                        system.monitor_symbols(symbols, interval, threshold)
                    except Exception as e:
                        print(f"Monitoring failed: {e}")
            elif choice == "4":
                print("\nStrategy Backtest")
                symbol = input("Enter symbol for backtest (e.g., AAPL): ").strip().upper()
                if symbol:
                    try:
                        days = int(input("Days to backtest (default 90): ") or "90")
                        result = system.run_backtest(symbol, days)
                        if not result:
                            print(f"Could not backtest {symbol}")
                    except Exception as e:
                        print(f"Backtest failed: {e}")
            elif choice == "5":
                print("\nGoodbye!")
                break
            else:
                print("Invalid choice. Please enter 1-5.")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
        except Exception as e:
            print(f"Unexpected error: {e}")
    print("System shutdown complete.")
