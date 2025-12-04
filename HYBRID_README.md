# Hybrid Advanced Trading System

A production-ready trading system that combines the best features from both the logistic multiclass system and the swing trading system.

## 🎯 Key Features

### From Logistic Multiclass System
- **Advanced Ensemble**: XGBoost + LightGBM + Random Forest
- **Isotonic Calibration**: Probability calibration for better predictions
- **Time-Series CV**: Proper temporal validation (no data leakage)
- **SMOTE/ADASYN**: Sophisticated class balancing
- **Strong Regularization**: Prevents overfitting with L1/L2, early stopping
- **Comprehensive Features**: 100+ technical indicators

### From Swing Trading System
- **Risk Management**: ATR-based stop-loss and take-profit levels
- **Real-Time Analysis**: Alpha Vantage API integration
- **Live Monitoring**: Multi-symbol watchlist
- **Backtesting**: Full trade simulation with exit reasons
- **Production Ready**: PyInstaller support, error handling
- **User-Friendly CLI**: Interactive command-line interface

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   HYBRID ENSEMBLE                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐         │
│  │ XGBoost  │  │ LightGBM │  │ Random Forest│         │
│  │  (35%)   │  │  (35%)   │  │    (30%)     │         │
│  └────┬─────┘  └────┬─────┘  └──────┬───────┘         │
│       │             │                │                  │
│       └─────────────┴────────────────┘                 │
│                     │                                   │
│              Weighted Average                           │
│                     │                                   │
│          Isotonic Calibration                          │
│                     │                                   │
│         3-Class Output (SHORT/PASS/LONG)               │
└─────────────────────────────────────────────────────────┘
                      │
         ┌────────────┴────────────┐
         │   Risk Management       │
         │  - ATR-based stops      │
         │  - Dynamic position     │
         │  - Risk/Reward ratio    │
         └─────────────────────────┘
```

## 🚀 Installation

### Required Dependencies

```bash
pip install numpy pandas scikit-learn xgboost lightgbm joblib requests ta

# Optional (for better class balancing)
pip install imbalanced-learn
```

### Optional: PyInstaller (for executable)

```bash
pip install pyinstaller
pyinstaller hybrid_trading_system.spec
```

## 📖 Usage

### 1. Training a Model

Train a short-term trading model (3-bar horizon):
```bash
python hybrid_trading_system.py \
    --mode train \
    --data ./historical_data \
    --trading-mode short_term \
    --model short_term_model.pkl \
    --cv-splits 5 \
    --balance smote_long_boost
```

Train a swing trading model (10-day horizon):
```bash
python hybrid_trading_system.py \
    --mode train \
    --data ./historical_data \
    --trading-mode swing \
    --model swing_model.pkl \
    --cv-splits 5 \
    --balance smote_balanced
```

### 2. Analyze Single Symbol

```bash
python hybrid_trading_system.py \
    --mode analyze \
    --symbol AAPL \
    --model hybrid_model.pkl \
    --api-key YOUR_API_KEY
```

Output example:
```
==============================================================
ANALYZING: AAPL
==============================================================

Timestamp: 2025-10-31 16:00:00
Price: $178.42
Change (1D): +1.23%
RSI: 58.3
Volume: 54,321,890

📊 PREDICTION: LONG
   Confidence: HIGH
   Probabilities:
     SHORT:  8.2% █
      PASS: 15.4% ███
      LONG: 76.4% ███████████████

💰 RISK MANAGEMENT:
   Entry: $178.42
   Stop Loss: $175.10 (-1.86%)
   Take Profit: $191.24 (+7.19%)
   Risk/Reward: 1:3.86

🔥 STRONG BUY SIGNAL!
==============================================================
```

### 3. Monitor Multiple Symbols

```bash
python hybrid_trading_system.py \
    --mode monitor \
    --symbols AAPL,MSFT,GOOGL,TSLA \
    --model hybrid_model.pkl \
    --api-key YOUR_API_KEY \
    --interval 300
```

This will continuously monitor the symbols and alert you when high-confidence opportunities are detected.

### 4. Backtest Strategy

```bash
python hybrid_trading_system.py \
    --mode backtest \
    --symbol AAPL \
    --days 90 \
    --model hybrid_model.pkl \
    --api-key YOUR_API_KEY
```

Output example:
```
==============================================================
BACKTEST RESULTS
==============================================================
Total Trades: 12
  LONG: 8
  SHORT: 4

Performance:
  Win Rate: 66.7%
  Avg Win: +5.3%
  Avg Loss: -2.1%
  Avg Profit/Trade: +2.4%
  Total Return: +31.2%
  Best Trade: +12.4%
  Worst Trade: -3.8%

Exit Reasons:
  Stop Loss:   3 trades (Win rate: 33.3%)
  Take Profit: 6 trades (Win rate: 100.0%)
  Max Time:    3 trades (Win rate: 66.7%)
==============================================================
```

## 📝 Command-Line Arguments

| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `--mode` | Operation mode | **Required** | `train`, `analyze`, `monitor`, `backtest` |
| `--data` | Training data directory | `./historical_data` | Any path |
| `--symbol` | Symbol for analyze/backtest | None | e.g., `AAPL` |
| `--symbols` | Symbols for monitoring | None | e.g., `AAPL,MSFT,GOOGL` |
| `--model` | Model file path | `hybrid_model.pkl` | Any `.pkl` file |
| `--api-key` | Alpha Vantage API key | `WEXRA3OHQYO9I592` | Your API key |
| `--trading-mode` | Trading strategy type | `swing` | `short_term`, `swing` |
| `--cv-splits` | Time-series CV splits | `5` | Any integer |
| `--balance` | Class balancing strategy | `smote_long_boost` | See below |
| `--interval` | Monitor interval (seconds) | `300` | Any integer |
| `--days` | Backtest period (days) | `90` | Any integer |

### Balance Strategies
- `smote_balanced`: Equal class representation
- `smote_long_boost`: 120% boost for LONG class (recommended)
- `smote_minority_boost`: Focus on minority classes
- `adasyn_balanced`: ADASYN with equal representation
- `adasyn_long_boost`: ADASYN with LONG boost
- `adasyn_minority_boost`: ADASYN minority focus

## 🔧 Technical Details

### Feature Engineering (100+ Features)

**Price Returns**
- Multiple timeframes: 1, 3, 5, 10, 20 bars

**Candlestick Patterns**
- Body size, shadows, ranges
- Doji, Hammer detection

**Moving Averages**
- SMA: 5, 7, 10, 20, 21, 50, 100, 200
- EMA: Same periods
- Crossovers and slopes

**Momentum Indicators**
- RSI (14, 21 periods)
- MACD (12, 26, 9)
- Stochastic Oscillator
- Williams %R
- ROC (10, 20)

**Volatility**
- ATR (14, 21)
- Bollinger Bands (20, 50)
- Rolling volatility
- Volatility ratios

**Trend Indicators**
- ADX (trend strength)
- CCI (commodity channel index)

**Volume**
- OBV (on-balance volume)
- Volume ratios
- Volume-weighted average price (VWAP)
- Volume price trend

**Time Features**
- Cyclical encoding (hour, day of week)
- Sin/Cos transformations

**Lag Features**
- 1, 2, 3, 5-period lags of key indicators

### Model Training

**XGBoost Parameters:**
- Learning rate: 0.005 (very conservative)
- Max depth: 4 (prevent overfitting)
- L1/L2 regularization: 5.0/10.0
- Early stopping: 300 rounds

**LightGBM Parameters:**
- Learning rate: 0.005
- Num leaves: 15
- L1/L2 regularization: 5.0/10.0
- Early stopping: 300 rounds

**Random Forest Parameters:**
- Estimators: 200
- Max depth: 10
- Class weights: Balanced

**Ensemble Weights:**
- XGBoost: 35%
- LightGBM: 35%
- Random Forest: 30%

### Risk Management

**Stop Loss Calculation:**
```
SL = max(entry_price * min_stop_pct, atr_value)
Risk/Reward Ratio: 0.5 (meaning 2:1 reward/risk)
```

**Take Profit Calculation:**
```
TP = max(entry_price * target_profit_pct, atr_value * 2)
Default target: 15%
```

## 📈 Performance Expectations

### Short-Term Mode (3-bar horizon)
- **Trade frequency**: High (multiple per day)
- **Hold period**: 1-3 bars (hours/days)
- **Target profit**: 0.5-2% per trade
- **Best for**: Active trading, crypto

### Swing Mode (10-day horizon)
- **Trade frequency**: Low (few per week)
- **Hold period**: 3-10 days
- **Target profit**: 5-15% per trade
- **Best for**: Stock swing trading

## ⚠️ Important Notes

1. **API Rate Limits**: Alpha Vantage free tier has rate limits. Add delays between requests.

2. **Data Quality**: Model performance depends on training data quality. Use clean, validated OHLCV data.

3. **Market Conditions**: The model is trained on historical data and may not perform well in unprecedented market conditions.

4. **Risk Disclaimer**: This is for educational purposes. Always practice proper risk management and never trade with money you can't afford to lose.

5. **Backtesting Bias**: Past performance does not guarantee future results. Backtest results may be optimistic due to look-ahead bias in indicator calculations.

## 🔄 Comparison with Original Systems

| Feature | Logistic System | Swing System | **Hybrid System** |
|---------|----------------|--------------|-------------------|
| Models | XGB + LGB | Random Forest | **XGB + LGB + RF** |
| Calibration | ✅ Isotonic | ❌ None | ✅ **Isotonic** |
| Validation | ✅ Time-Series CV | ❌ Simple split | ✅ **Time-Series CV** |
| Risk Management | ❌ None | ✅ ATR-based | ✅ **ATR-based** |
| Real-time | ❌ No | ✅ Yes | ✅ **Yes** |
| Backtesting | ❌ No | ✅ Yes | ✅ **Yes** |
| Features | ~40 | ~60 | **~100** |
| Production Ready | ❌ No | ✅ Yes | ✅ **Yes** |

## 🛠️ Troubleshooting

### ImportError: No module named 'imblearn'
```bash
pip install imbalanced-learn
```
The system will still work without it, but class balancing will be disabled.

### API Rate Limit Errors
- Use `--interval` to increase time between requests
- Get a premium Alpha Vantage API key
- Use cached/downloaded historical data for backtesting

### Low Model Performance
- Increase training data size (need 500+ samples minimum)
- Adjust `--balance` strategy
- Try different `--trading-mode`
- Check for data quality issues

### Memory Issues
- Reduce number of features
- Use smaller training datasets
- Reduce `--cv-splits`

## 📚 Data Format

Training data should be CSV files with these columns:
- `Date` or `timestamp`: Datetime
- `open`: Opening price
- `high`: High price
- `low`: Low price
- `close`: Closing price
- `volume`: Trading volume
- `adj_close` (optional): Adjusted closing price

Example:
```csv
Date,open,high,low,close,volume
2025-01-01,100.0,102.5,99.5,101.0,1000000
2025-01-02,101.0,103.0,100.0,102.5,1200000
```

## 🤝 Contributing

Improvements welcome! Consider:
- Additional feature engineering
- More sophisticated risk management
- Alternative data sources
- Enhanced backtesting metrics
- UI improvements

## 📄 License

MIT License - Use at your own risk

## 🙏 Credits

Combined and enhanced from:
- Logistic multiclass trading system (ensemble ML architecture)
- Swing trading prediction bot (production features)
- Technical Analysis Library (ta-lib)

---

**Built with ❤️ for algorithmic traders**
