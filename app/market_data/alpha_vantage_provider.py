"""US market (and select commodities) data via Alpha Vantage."""
import pandas as pd
import requests

from app.config import ALPHA_VANTAGE_MIN_INTERVAL_SECONDS
from app.market_data.base import MarketDataProvider
from app.security import SecurityValidator

_COMMODITY_MAP = {
    "COPPER": "COPPER",
    "ALUMINUM": "ALUMINUM",
    "AL": "ALUMINUM",
    "GOLD": "XAU",
    "XAU": "XAU",
    "SILVER": "XAG",
    "XAG": "XAG",
}

_TIME_SERIES_KEYS = [
    "Time Series (Daily)",
    "Time Series (60min)",
    "Time Series (5min)",
    "Weekly Adjusted Time Series",
    "Monthly Adjusted Time Series",
]


class AlphaVantageProvider(MarketDataProvider):
    BASE_URL = "https://www.alphavantage.co/query?"

    def __init__(self, api_key=None):
        super().__init__(min_request_interval_seconds=ALPHA_VANTAGE_MIN_INTERVAL_SECONDS)
        self.api_key = SecurityValidator.sanitize_text(api_key, "ALPHA_VANTAGE_API_KEY", 128) if api_key else None

    def _get(self, params, timeout=30):
        if not self.api_key:
            raise ValueError(
                "ALPHA_VANTAGE_API_KEY is required for US market requests. "
                "Korean market requests use KRX_SERVICE_KEY from data.go.kr."
            )
        response = self._paced_get(self.BASE_URL, {**params, "apikey": self.api_key}, timeout=timeout)
        response.raise_for_status()
        return response

    def _fetch_commodity(self, symbol, commodity_func, interval):
        params = {"function": commodity_func, "interval": interval}
        response = self._get(params, timeout=30)
        data = response.json()
        self._raise_for_api_errors(data)
        if "data" not in data:
            raise ValueError("No commodity data found in API response")

        rows = []
        for row in data["data"]:
            try:
                value = float(row.get("value", 0))
                rows.append({
                    "timestamp": pd.to_datetime(row["date"]),
                    "open": value, "high": value, "low": value, "close": value, "volume": 0,
                })
            except (TypeError, ValueError, KeyError):
                continue

        df = pd.DataFrame(rows)
        if df.empty:
            print(f"No data returned for commodity {symbol}")
            return None
        df = df.set_index("timestamp")
        return self.finalize_ohlcv_frame(df)

    def _fetch_time_series(self, symbol, function, outputsize):
        params = {"function": function, "symbol": symbol, "outputsize": outputsize}
        response = self._get(params, timeout=30)
        data = response.json()

        if outputsize == "full" and "Information" in data and "premium" in data["Information"].lower():
            print(f"outputsize=full is a premium-only feature for {symbol}; falling back to outputsize=compact.")
            params["outputsize"] = "compact"
            response = self._get(params, timeout=30)
            data = response.json()

        self._raise_for_api_errors(data)

        time_series_data = next((data[key] for key in _TIME_SERIES_KEYS if key in data), None)
        if time_series_data is None:
            raise ValueError("No time series data found in API response")

        rows = []
        for timestamp, values in time_series_data.items():
            rows.append({
                "timestamp": pd.to_datetime(timestamp),
                "open": float(values.get("1. open", 0)),
                "high": float(values.get("2. high", 0)),
                "low": float(values.get("3. low", 0)),
                "close": float(values.get("4. close", 0)),
                "volume": int(float(values.get("5. volume", 0))),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return None
        df = df.set_index("timestamp")
        return self.finalize_ohlcv_frame(df)

    @staticmethod
    def _raise_for_api_errors(data):
        if "Error Message" in data:
            raise ValueError(f"Alpha Vantage API Error: {data['Error Message']}")
        if "Note" in data:
            raise ValueError(f"Alpha Vantage API Rate Limit: {data['Note']}")
        if "Information" in data:
            raise ValueError(f"Alpha Vantage API Info: {data['Information']}")

    def fetch_ohlcv(self, symbol, function="TIME_SERIES_DAILY", outputsize="compact", interval="monthly", **kwargs):
        try:
            if not self.api_key:
                print(
                    "US market data is disabled because ALPHA_VANTAGE_API_KEY is not set. "
                    "Korean market data continues to use data.go.kr via KRX_SERVICE_KEY."
                )
                return None

            symbol_upper = symbol.upper()
            if symbol_upper in _COMMODITY_MAP:
                df = self._fetch_commodity(symbol, _COMMODITY_MAP[symbol_upper], interval)
            else:
                df = self._fetch_time_series(symbol, function, outputsize)

            if df is not None:
                print(f"Fetched {len(df)} data points for {symbol}")
            return df
        except requests.exceptions.RequestException as error:
            print(f"Network error fetching data for {symbol}: {error}")
            return None
        except Exception as error:
            print(f"Error fetching Alpha Vantage data for {symbol}: {error}")
            return None

    def resolve_company_name(self, symbol):
        try:
            response = self._get({"function": "OVERVIEW", "symbol": symbol}, timeout=15)
            name = response.json().get("Name")
            if name:
                return name
        except Exception:
            pass

        try:
            response = self._get({"function": "SYMBOL_SEARCH", "keywords": symbol}, timeout=15)
            best_matches = response.json().get("bestMatches") or []
            for match in best_matches:
                if match.get("1. symbol", "").upper() == symbol.upper():
                    return match.get("2. name") or symbol
            if best_matches:
                return best_matches[0].get("2. name") or symbol
        except Exception:
            pass

        return symbol
