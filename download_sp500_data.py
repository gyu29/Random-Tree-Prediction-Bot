import yfinance as yf
import pandas as pd
from datetime import datetime
import os

def download_sp500_data():
    data_dir = "./historical_data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    start_date = "1990-01-01"
    end_date = "2025-07-27"
    
    print(f"Downloading S&P 500 data from {start_date} to {end_date}")
    
    try:
        ticker = yf.Ticker("^GSPC")
        df = ticker.history(start=start_date, end=end_date)
        
        if df.empty:
            raise ValueError("No data was downloaded")
            
        output_file = os.path.join(data_dir, "sp500_historical.csv")
        df.to_csv(output_file)
        
        print(f"Data downloaded successfully!")
        print(f"Total records: {len(df)}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Data saved to: {output_file}")
        
    except Exception as e:
        print(f"Error downloading data: {e}")

if __name__ == "__main__":
    download_sp500_data()
