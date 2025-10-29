# Random Tree Prediction Bot

A machine learning-powered swing trading system that uses Random Forest algorithms to predict stock market trends and generate trading signals for S&P 500 stocks.

## Overview

This project combines advanced machine learning techniques with financial market analysis to create an automated swing trading prediction system. The system downloads historical S&P 500 data, trains a Random Forest model on technical indicators, and provides an interactive dashboard for viewing predictions and trading signals.

## Features

- **Automated Data Collection**: Downloads and processes historical S&P 500 stock data
- **Machine Learning Predictions**: Uses enhanced Random Forest models for swing trading predictions
- **Technical Analysis**: Incorporates multiple technical indicators and features
- **Interactive Dashboard**: Web-based interface built with Vite for viewing predictions and analysis
- **Model Persistence**: Saves trained models, scalers, and feature configurations for reuse
- **Standalone Executable**: Can be packaged as a standalone application using PyInstaller

## Project Structure

```
Random-Tree-Prediction-Bot/
├── historical_data/          # Historical stock data storage
├── swing-dashboard/          # Frontend dashboard (Vite + JavaScript)
├── download_sp500_[data.py](http://data.py)    # Data collection script
├── swing_trading_[system.py](http://system.py)   # Main trading system logic
├── swing_model_enhanced.pkl  # Trained Random Forest model
├── swing_scaler_enhanced.pkl # Feature scaler
├── feature_columns_enhanced.pkl # Feature configuration
├── training_stats.pkl        # Model training statistics
├── sp500_historical.csv      # S&P 500 historical data
├── vite.config.js            # Vite configuration
└── *.spec files              # PyInstaller build configurations
```

## Technologies Used

**Backend:**

- Python (51.2%)
- scikit-learn for Random Forest implementation
- pandas for data manipulation
- scipy for statistical analysis

**Frontend:**

- JavaScript (46.3%)
- Vite for build tooling
- CSS (1.8%)
- HTML (0.7%)

**Deployment:**

- PyInstaller for creating standalone executables

## Installation

### Prerequisites

- Python 3.x
- Node.js and npm (for dashboard)
- pip for Python package management

### Setup

1. Clone the repository:

```bash
git clone https://github.com/gyu29/Random-Tree-Prediction-Bot.git
cd Random-Tree-Prediction-Bot
```

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

1. Install dashboard dependencies:

```bash
cd swing-dashboard
npm install
cd ..
```

## Usage

### Download Historical Data

```bash
python download_sp500_[data.py](http://data.py)
```

### Run the Trading System

```bash
python swing_trading_[system.py](http://system.py)
```

### Launch the Dashboard

```bash
cd swing-dashboard
npm run dev
```

### Build Standalone Executable

```bash
pyinstaller SwingTradingSystem.spec
```

## How It Works

1. **Data Collection**: The system downloads historical price and volume data for S&P 500 stocks
2. **Feature Engineering**: Technical indicators and features are calculated and normalized
3. **Model Training**: A Random Forest classifier is trained on historical patterns
4. **Prediction**: The model generates swing trading signals (buy/sell/hold)
5. **Visualization**: Results are displayed in an interactive web dashboard

## Model Details

The system uses an enhanced Random Forest model with:

- Optimized hyperparameters for swing trading
- Feature scaling for improved prediction accuracy
- Multiple technical indicators as input features
- Persistent model storage for quick deployment

## Dashboard Features

- Real-time prediction visualization
- Historical performance metrics
- Interactive stock charts
- Trading signal indicators
- Model statistics and accuracy metrics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer

**This software is for educational and research purposes only. It is not financial advice. Trading stocks involves risk, and you should never trade with money you cannot afford to lose. Past performance does not guarantee future results.**

## License

Please refer to the repository for license information.

## Author

**gyu29**

- GitHub: [@gyu29](https://github.com/gyu29)

---

⭐ If you find this project helpful, please consider giving it a star!
