import yfinance as yf
import pandas as pd
from datetime import datetime
import os

def _safe_ticker_filename(ticker):
    return ticker.replace("^", "").replace("/", "_").replace("\\", "_").strip()


def download_sp500_data(tickers):
    data_dir = "./historical_data"
    os.makedirs(data_dir, exist_ok=True)

    start_date = "1990-01-01"
    end_date = "2025-07-27"

    if not isinstance(tickers, list) or not tickers:
        raise ValueError("tickers must be a non-empty list")

    print(f"Downloading {len(tickers)} ticker(s) from {start_date} to {end_date}")

    for ticker_symbol in tickers:
        try:
            ticker_symbol = str(ticker_symbol).strip()
            if not ticker_symbol:
                continue

            print(f"\nDownloading: {ticker_symbol}")
            ticker = yf.Ticker(ticker_symbol)
            df = ticker.history(start=start_date, end=end_date)

            if df.empty:
                print(f"No data downloaded for {ticker_symbol}")
                continue

            safe_name = _safe_ticker_filename(ticker_symbol)
            output_file = os.path.join(data_dir, f"{safe_name}_historical.csv")
            df.to_csv(output_file)

            print(f"Data downloaded successfully for {ticker_symbol}!")
            print(f"Total records: {len(df)}")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            print(f"Data saved to: {output_file}")

        except Exception as e:
            print(f"Error downloading data for {ticker_symbol}: {e}")

if __name__ == "__main__":
    tickers_to_download = [
        "^GSPC",  # S&P 500
        "AAPL",   # Apple Inc.
        "MSFT",   # Microsoft Corporation
        "GOOGL",  # Alphabet Inc.
        "AMZN",   # Amazon.com, Inc.
        "META",   # Meta Platforms, Inc.
        "TSLA",   # Tesla, Inc.
        "BRK-B",  # Berkshire Hathaway Inc. (Class B)
        "JNJ",    # Johnson & Johnson
        "V",      # Visa Inc.
        "WMT",    # Walmart Inc.
        "PG",     # Procter & Gamble Co.   
        "JPM",    # JPMorgan Chase & Co.
        "NVDA",   # NVIDIA Corporation
        "DIS",    # The Walt Disney Company
        "HD",     # The Home Depot, Inc.
        "MA",     # Mastercard Incorporated
        "BAC",    # Bank of America Corporation
        "XOM",    # Exxon Mobil Corporation
        "VZ",     # Verizon Communications Inc.
        "ADBE",   # Adobe Inc.
        "NFLX",   # Netflix, Inc.
        "PYPL",   # PayPal Holdings, Inc.
        "INTC",   # Intel Corporation
        "CSCO",   # Cisco Systems, Inc.
        "PFE",    # Pfizer Inc.
        "TSMC",   # Taiwan Semiconductor Manufacturing Company Limited
        "UBER",   # Uber Technologies, Inc.
        "NET",    # Cloudflare, Inc.
        "ZM",     # Zoom Video Communications, Inc.
        "UNH",    # UnitedHealth Group Incorporated
        "CVX",    # Chevron Corporation
        "USD",   # US Dollar Index
        "BTC-USD",  # Bitcoin
        "USOI",   # US Oil Fund
        "GLD",    # SPDR Gold Shares
        "SILVER",  # iShares Silver Trust
        "VIX",    # CBOE Volatility Index
        "NDQA",   # Nasdaq 100 Index
        "DJI",    # Dow Jones Industrial Average
    ]
    download_sp500_data(tickers_to_download)
