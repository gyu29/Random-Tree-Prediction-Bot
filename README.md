# Random-Tree-Prediction-Bot
# Advanced Swing Trading ML System

A comprehensive machine learning-based swing trading system that uses technical indicators and Random Forest classification to identify profitable swing trade opportunities in financial markets.

## Features

- **Machine Learning Model**: Random Forest classifier with imbalanced class handling
- **Technical Analysis**: 100+ technical indicators including RSI, MACD, Bollinger Bands, ATR, and more
- **Real-time Analysis**: Live market data integration via Alpha Vantage API
- **Risk Management**: Automatic stop-loss and take-profit calculation based on ATR
- **Backtesting**: Historical performance testing with detailed metrics
- **Multi-symbol Monitoring**: Simultaneous analysis of multiple stocks
- **Data Processing**: Robust CSV/Excel data loading and preprocessing

## 📋 Requirements

```
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
ta>=0.8.0
joblib>=1.0.0
requests>=2.25.0
```

## 🛠️ Installation

1. Clone or download the system files
2. Install required dependencies:
```bash
pip install pandas numpy scikit-learn ta joblib requests
```

3. Get a free Alpha Vantage API key from [alphavantage.co](https://www.alphavantage.co/support/#api-key)

4. Update the API key in the script:
```python
ALPHA_VANTAGE_API_KEY = "YOUR_API_KEY_HERE"
```

## 📁 Data Structure

### Historical Data Directory
Create a `./historical_data/` directory and place your historical stock data files in supported formats:
- CSV files (*.csv)
- Excel files (*.xlsx)
- Parquet files (*.parquet)

### Required Data Columns
Your data files must contain these columns (case-insensitive):
- `Date` or datetime index
- `Open`
- `High` 
- `Low`
- `Close`
- `Volume`

Optional columns:
- `Adj Close` (adjusted close price)
- `Dividends`
- `Stock Splits`

## 🎯 Usage

### Command Line Interface

Run the system:
```bash
python swing_trading_system.py
```

The interactive menu provides 5 options:

#### 1. Train New Model
- Processes historical data from `./historical_data/` directory
- Creates technical indicators
- Trains Random Forest classifier
- Saves model files for future use

**Parameters:**
- `swing_threshold`: Minimum price movement to consider a swing (default: 0.15 = 15%)
- `lookforward_periods`: Days to look ahead for swing detection (default: 10)

#### 2. Analyze Single Symbol
- Fetches real-time data for a specific stock
- Generates buy/sell recommendation
- Provides probability scores and confidence levels
- Calculates stop-loss and take-profit levels

#### 3. Monitor Multiple Symbols
- Continuously monitors multiple stocks
- Alerts when high-probability opportunities are found
- Customizable check intervals and alert thresholds

#### 4. Run Backtest
- Tests strategy performance on historical data
- Provides win rate, average profit, and trade statistics
- Shows performance by exit reason (stop-loss, take-profit, max time)

#### 5. Exit
- Safely shuts down the system

### Programmatic Usage

```python
from swing_trading_system import SwingTradingSystem

# Initialize system
system = SwingTradingSystem(api_key="YOUR_API_KEY")

# Train model
system.train_model(
    data_directory="./historical_data",
    swing_threshold=0.15,
    lookforward_periods=10
)

# Analyze a symbol
result = system.analyze_symbol("AAPL")

# Run backtest
backtest_results = system.run_backtest("AAPL", days_back=90)

# Monitor symbols
system.monitor_symbols(["AAPL", "MSFT", "GOOGL"], check_interval=300)
```

## 📊 Model Output

### Analysis Results
```
SWING TRADE ANALYSIS: AAPL
Current Price: $175.23
Swing Probability: 85.2%
Confidence Level: High
🚨 SWING TRADE OPPORTUNITY DETECTED! 🚨
Recommended Stop-Loss: $168.45
Recommended Take-Profit: $184.67
```

### Backtest Results
```
BACKTEST RESULTS: AAPL
Total Trades: 12
Win Rate: 66.7%
Average Profit per Trade: 2.34%
Total Return: 31.2%
Stop-Loss Exits: 3 (Win Rate: 0.0%)
Take-Profit Exits: 5 (Win Rate: 100.0%)
Max Time Exits: 4 (Win Rate: 75.0%)
```

## 🧠 Technical Indicators

The system creates 100+ technical indicators including:

### Trend Indicators
- Simple Moving Averages (5, 10, 20, 50, 100, 200 periods)
- Exponential Moving Averages (5, 10, 20, 50, 100, 200 periods)
- MACD (12/26/9)
- ADX (Average Directional Index)
- CCI (Commodity Channel Index)

### Momentum Indicators
- RSI (14, 21 periods)
- Williams %R
- Stochastic Oscillator
- Rate of Change (10, 20 periods)

### Volatility Indicators
- Bollinger Bands (20, 50 periods)
- Average True Range (14, 21 periods)
- Volatility ratios and squeeze indicators

### Volume Indicators
- Volume Price Trend
- On Balance Volume
- Volume Weighted Average Price
- Volume ratios

### Price Pattern Recognition
- Doji patterns
- Hammer patterns
- Body size analysis
- Shadow analysis

## ⚙️ Configuration

### Model Parameters
```python
# Random Forest Configuration
n_estimators = 100
max_depth = 8
min_samples_split = 20
min_samples_leaf = 10
class_weight = {0: 1, 1: 50}  # Heavy weight for swing opportunities
```

### Risk Management
```python
# Stop-loss and take-profit calculation
risk_reward_ratio = 0.5
minimum_stop_loss = 1% or 1 ATR
take_profit = max(swing_threshold, 2 * ATR)
```

## 📈 Performance Optimization

### For Better Results:
1. **More Data**: Use at least 2-3 years of historical data
2. **Quality Data**: Ensure clean, complete OHLCV data
3. **Parameter Tuning**: Adjust swing_threshold based on market conditions
4. **Regular Retraining**: Retrain model monthly with new data
5. **Risk Management**: Always use stop-losses and position sizing

### Handling Class Imbalance:
- The system uses class weights to handle rare swing opportunities
- Creates synthetic positive samples when needed
- Uses out-of-bag scoring for better validation

## 🔧 Troubleshooting

### Common Issues:

**"Insufficient data" error:**
- Ensure you have at least 500 data points for training
- Check that your data files have the required columns

**"Missing required columns" error:**
- Verify column names match expected format
- The system will attempt to standardize column names automatically

**API rate limiting:**
- Free Alpha Vantage API has rate limits (5 calls/minute, 500 calls/day)
- Add delays between API calls for multiple symbol monitoring

**Low swing detection rate:**
- Try lowering the swing_threshold parameter
- Increase lookforward_periods for longer-term swings
- Ensure sufficient historical data for training

## 📝 Model Files

After training, the system saves:
- `swing_model_enhanced.pkl` - Trained Random Forest model
- `swing_scaler_enhanced.pkl` - Feature scaler
- `feature_columns_enhanced.pkl` - Required feature list
- `training_stats.pkl` - Training performance metrics

## ⚠️ Disclaimer

This system is for educational and research purposes only. Past performance does not guarantee future results. Always conduct your own research and consider your risk tolerance before making any trading decisions. The authors are not responsible for any financial losses incurred from using this system.

## 🤝 Contributing

Feel free to submit issues, feature requests, or improvements to enhance the system's functionality.

## 📄 License

This project is provided as-is for educational purposes. Please review and comply with your local financial regulations before using for live trading.
