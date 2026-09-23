import abc
import datetime
import logging
import time
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

TROY_OUNCE_TO_GRAMS = 31.1034768

class MarketDataProvider(abc.ABC):
    @abc.abstractmethod
    def get_historical_prices(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        ticker: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical price data for a symbol.
        Returns a DataFrame with columns: [timestamp, open, high, low, close, volume, provider, source, retrieval_timestamp, instrument, market, currency, unit, timezone]
        """
        pass

class YFinanceProvider(MarketDataProvider):
    def __init__(self, retries: int = 3, backoff_factor: float = 1.5, timeout: int = 15):
        self.retries = retries
        self.backoff_factor = backoff_factor
        self.timeout = timeout

    def get_historical_prices(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        ticker: Optional[str] = None
    ) -> pd.DataFrame:
        import yfinance as yf

        target_ticker = ticker if ticker else symbol
        retrieval_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

        for attempt in range(1, self.retries + 1):
            try:
                logger.info(f"Fetching {symbol} ({target_ticker}) attempt {attempt}")
                df = yf.download(
                    target_ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    timeout=self.timeout
                )
                if df.empty:
                    raise ValueError(f"Empty data returned for ticker {target_ticker}")

                # Flatten multi-index columns if present (yfinance 0.2.x+ behavior)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df = df.reset_index()
                # Rename columns standard lower case
                rename_map = {
                    "Date": "timestamp",
                    "Datetime": "timestamp",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Adj Close": "adj_close",
                    "Volume": "volume"
                }
                df = df.rename(columns=rename_map)

                df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d")
                df["provider"] = "yfinance"
                df["source"] = target_ticker
                df["retrieval_timestamp"] = retrieval_ts
                df["instrument"] = symbol
                df["timezone"] = "UTC"

                # Clean up numeric columns
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    else:
                        df[col] = np.nan

                return df[["timestamp", "open", "high", "low", "close", "volume", "provider", "source", "retrieval_timestamp", "instrument", "timezone"]]

            except Exception as e:
                logger.warning(f"Error fetching {target_ticker} on attempt {attempt}: {e}")
                if attempt == self.retries:
                    logger.error(f"Failed to fetch {target_ticker} after {self.retries} attempts.")
                    raise e
                time.sleep(self.backoff_factor ** attempt)


class SyntheticProvider(MarketDataProvider):
    """
    Deterministic synthetic market data provider for offline testing and offline pipelines.
    Generates realistic historical price series with controllable seed.
    """
    def __init__(self, seed: int = 42):
        self.seed = seed

    def get_historical_prices(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        ticker: Optional[str] = None
    ) -> pd.DataFrame:
        np.random.seed(self.seed + abs(hash(symbol)) % 100000)
        dates = pd.date_range(start=start_date, end=end_date, freq="B")

        base_prices = {
            "GOLD": 2000.0,
            "SILVER": 24.0,
            "GOLD_FUTURES": 2010.0,
            "SILVER_FUTURES": 24.2,
            "USDINR": 83.0,
            "DXY": 104.0,
            "US10Y": 4.2,
            "US2Y": 4.5,
            "US_REAL_YIELD": 1.8,
            "CRUDE_OIL": 75.0,
            "SP500": 5000.0,
            "NIFTY50": 22000.0,
            "VIX": 15.0,
            "INDIA_VIX": 14.0,
            "MCX_GOLD": 65000.0,
            "MCX_SILVER": 75000.0
        }

        base_price = base_prices.get(symbol, 100.0)
        num_days = len(dates)

        # Geometric Brownian Motion with slight upward drift
        returns = np.random.normal(0.0003, 0.012, num_days)
        price_path = base_price * np.exp(np.cumsum(returns))

        records = []
        retrieval_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

        for i, date in enumerate(dates):
            close = float(price_path[i])
            daily_vol = close * 0.008
            open_p = close + np.random.normal(0, daily_vol)
            high_p = max(open_p, close) + abs(np.random.normal(0, daily_vol))
            low_p = min(open_p, close) - abs(np.random.normal(0, daily_vol))
            volume = float(np.random.randint(10000, 500000))

            records.append({
                "timestamp": date.strftime("%Y-%m-%d"),
                "open": round(open_p, 4),
                "high": round(high_p, 4),
                "low": round(low_p, 4),
                "close": round(close, 4),
                "volume": volume,
                "provider": "synthetic",
                "source": "simulation",
                "retrieval_timestamp": retrieval_ts,
                "instrument": symbol,
                "timezone": "UTC"
            })

        return pd.DataFrame(records)


def convert_gold_usd_oz_to_inr_10g(usd_price_per_oz: float, usdinr_rate: float) -> float:
    """
    Convert Gold price from USD/oz to INR/10g.
    1 troy oz = 31.1034768 grams.
    Price per gram in USD = usd_price_per_oz / 31.1034768
    Price per gram in INR = (usd_price_per_oz / 31.1034768) * usdinr_rate
    Price per 10 grams in INR = ((usd_price_per_oz / 31.1034768) * usdinr_rate) * 10
    """
    if pd.isna(usd_price_per_oz) or pd.isna(usdinr_rate) or usd_price_per_oz <= 0 or usdinr_rate <= 0:
        return np.nan
    return (usd_price_per_oz / TROY_OUNCE_TO_GRAMS) * usdinr_rate * 10.0


def convert_silver_usd_oz_to_inr_kg(usd_price_per_oz: float, usdinr_rate: float) -> float:
    """
    Convert Silver price from USD/oz to INR/kg.
    1 troy oz = 31.1034768 grams = 0.0311034768 kg.
    Price per kg in USD = usd_price_per_oz / 0.0311034768
    Price per kg in INR = (usd_price_per_oz / 0.0311034768) * usdinr_rate
    """
    if pd.isna(usd_price_per_oz) or pd.isna(usdinr_rate) or usd_price_per_oz <= 0 or usdinr_rate <= 0:
        return np.nan
    return (usd_price_per_oz / (TROY_OUNCE_TO_GRAMS / 1000.0)) * usdinr_rate
