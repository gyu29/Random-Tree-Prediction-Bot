# Quick Start Guide - Hybrid Trading System

## 🚀 5-Minute Setup

### Step 1: Install Dependencies

```bash
# Core dependencies (required)
pip install numpy pandas scikit-learn xgboost lightgbm joblib requests ta

# Optional (for better performance)
pip install imbalanced-learn
```

### Step 2: Prepare Training Data

Place your CSV files in `./historical_data/` directory:
```
historical_data/
├── sp500_historical.csv
├── aapl_data.csv
└── msft_data.csv
```

### Step 3: Train Your First Model

```bash
# Train a swing trading model (recommended for beginners)
python hybrid_trading_system.py \
    --mode train \
    --data ./historical_data \
    --trading-mode swing \
    --model my_first_model.pkl
```

Wait 5-15 minutes for training to complete.

### Step 4: Analyze a Stock

```bash
# Analyze Apple stock
python hybrid_trading_system.py \
    --mode analyze \
    --symbol AAPL \
    --model my_first_model.pkl \
    --api-key YOUR_ALPHAVANTAGE_KEY
```

## 📊 Example Workflows

### Workflow 1: Day Trading (Short-term)

```bash
# 1. Train short-term model
python hybrid_trading_system.py \
    --mode train \
    --trading-mode short_term \
    --model day_trading.pkl

# 2. Monitor tech stocks every 5 minutes
python hybrid_trading_system.py \
    --mode monitor \
    --symbols AAPL,MSFT,GOOGL,AMZN,TSLA \
    --model day_trading.pkl \
    --interval 300
```

### Workflow 2: Swing Trading (Medium-term)

```bash
# 1. Train swing model
python hybrid_trading_system.py \
    --mode train \
    --trading-mode swing \
    --model swing_trading.pkl

# 2. Find opportunities
python hybrid_trading_system.py \
    --mode analyze \
    --symbol NVDA \
    --model swing_trading.pkl

# 3. Backtest before using
python hybrid_trading_system.py \
    --mode backtest \
    --symbol NVDA \
    --days 180 \
    --model swing_trading.pkl
```

### Workflow 3: Portfolio Scanner

```bash
# Monitor your entire portfolio
python hybrid_trading_system.py \
    --mode monitor \
    --symbols AAPL,MSFT,GOOGL,AMZN,META,TSLA,NVDA,AMD,NFLX,DIS \
    --model swing_trading.pkl \
    --interval 600
```

## 🎯 Best Practices

### 1. Start with Swing Trading
- Less sensitive to noise
- Better risk/reward ratios
- Easier to validate

### 2. Backtest Everything
```bash
# Always backtest before using a model
python hybrid_trading_system.py --mode backtest --symbol YOUR_SYMBOL --model YOUR_MODEL.pkl
```

### 3. Use Multiple Models
Train different models for different market conditions:
- `bull_market_model.pkl` (high risk/reward)
- `bear_market_model.pkl` (conservative)
- `volatile_model.pkl` (tight stops)

### 4. Monitor Key Metrics
When training, look for:
- ✅ Win rate > 60%
- ✅ Test LogLoss < 0.5
- ✅ Balanced class predictions
- ⚠️ Train/Val loss gap < 0.05

### 5. Paper Trade First
- Never use real money without paper trading
- Track predictions vs reality
- Adjust thresholds based on results

## ⚙️ Configuration Examples

### Conservative Trading (Low Risk)
```bash
python hybrid_trading_system.py \
    --mode train \
    --trading-mode swing \
    --balance smote_balanced \
    --cv-splits 5
```
- Equal class representation
- More validation splits
- Longer holding periods

### Aggressive Trading (High Risk)
```bash
python hybrid_trading_system.py \
    --mode train \
    --trading-mode short_term \
    --balance smote_long_boost \
    --cv-splits 3
```
- Boost LONG signals
- Faster trading
- Higher frequency

## 🔍 Reading the Output

### Training Output
```
Total samples: 5000
  SHORT: 800 (16.0%)
  PASS: 3200 (64.0%)
  LONG: 1000 (20.0%)

After balancing: 9600 samples
Balanced %: SHORT=33.3%, PASS=33.3%, LONG=33.3%

[1/3] Training XGBoost...
  Trained at iteration 2341
  Train loss: 0.4523, Val loss: 0.4892

TEST SET EVALUATION
              precision    recall  f1-score   support
       SHORT     0.7234    0.6891    0.7058       160
        PASS     0.8123    0.8456    0.8286       640
        LONG     0.7456    0.7123    0.7286       200

Log Loss: 0.4892
```

**Good signs:**
- ✅ Balanced distribution after SMOTE
- ✅ Val loss close to train loss
- ✅ F1-scores > 0.70

**Bad signs:**
- ⚠️ Val loss >> train loss (overfitting)
- ⚠️ Very low recall for LONG/SHORT
- ⚠️ LogLoss > 1.0

### Analysis Output
```
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
```

**Interpretation:**
- **HIGH confidence + LONG**: Strong buy signal
- **Risk/Reward > 2.0**: Good opportunity
- **Stop loss -1.86%**: Acceptable risk

## 🐛 Common Issues

### Issue: "Insufficient data"
**Solution:** Need at least 500 rows of OHLCV data
```bash
# Check your data
python -c "import pandas as pd; df = pd.read_csv('historical_data/data.csv'); print(len(df))"
```

### Issue: "No models available"
**Solution:** XGBoost or LightGBM not installed
```bash
pip install xgboost lightgbm
```

### Issue: "API rate limit"
**Solution:** Increase interval or get premium key
```bash
# Increase to 10 minutes
python hybrid_trading_system.py --mode monitor --symbols AAPL --interval 600
```

### Issue: "Poor model performance"
**Solutions:**
1. More training data
2. Better data quality
3. Different balance strategy
4. Tune hyperparameters

## 📚 Next Steps

1. **Read the full README**: `HYBRID_README.md`
2. **Experiment with parameters**: Try different configurations
3. **Build your watchlist**: Find symbols that match your strategy
4. **Track performance**: Keep a trading journal
5. **Iterate and improve**: Refine based on results

## 💡 Pro Tips

1. **Use cron for scheduled analysis**
```bash
# Add to crontab (runs every 30 minutes during market hours)
*/30 9-16 * * 1-5 cd /path/to/project && python hybrid_trading_system.py --mode monitor --symbols AAPL,MSFT >> monitor.log 2>&1
```

2. **Create symbol lists**
```bash
# Create a file: watchlist.txt
AAPL
MSFT
GOOGL
AMZN

# Monitor from file
python hybrid_trading_system.py --mode monitor --symbols $(cat watchlist.txt | tr '\n' ',')
```

3. **Save results to CSV**
```bash
python hybrid_trading_system.py --mode backtest --symbol AAPL > backtest_results.txt
```

4. **Compare models**
```bash
# Train multiple models
python hybrid_trading_system.py --mode train --model model_v1.pkl
python hybrid_trading_system.py --mode train --balance smote_balanced --model model_v2.pkl

# Test both
python hybrid_trading_system.py --mode backtest --symbol AAPL --model model_v1.pkl
python hybrid_trading_system.py --mode backtest --symbol AAPL --model model_v2.pkl
```

## 🆘 Get Help

If you encounter issues:
1. Check the error message carefully
2. Verify data format matches requirements
3. Ensure all dependencies are installed
4. Try with a single symbol first
5. Use `--help` for all options

```bash
python hybrid_trading_system.py --help
```

---

**Happy Trading! 🚀📈**

Remember: This is a tool to assist decision-making, not a guarantee of profits. Always do your own research and practice proper risk management.
