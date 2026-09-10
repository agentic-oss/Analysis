import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

class MarketDataProvider(ABC):
    """Abstract base class for market data providers."""

    @abstractmethod
    def get_historical_prices(
        self,
        symbol: str,
        ticker: str,
        start_date: str,
        end_date: str,
        instrument_meta: Dict[str, Any]
    ) -> pd.DataFrame:
        """Fetch historical price data for a symbol.
        Returns DataFrame with columns: date, open, high, low, close, volume, symbol, provider, currency, unit, timezone, retrieval_timestamp
        """
        pass

    @abstractmethod
    def get_economic_events(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch economic calendar/event data."""
        pass


class YFinanceMarketDataProvider(MarketDataProvider):
    """yfinance implementation of MarketDataProvider with retries and exponential backoff."""

    def __init__(self, retries: int = 3, timeout: int = 10, backoff_factor: float = 1.5):
        self.retries = retries
        self.timeout = timeout
        self.backoff_factor = backoff_factor

    def get_historical_prices(
        self,
        symbol: str,
        ticker: str,
        start_date: str,
        end_date: str,
        instrument_meta: Dict[str, Any]
    ) -> pd.DataFrame:
        attempt = 0
        delay = 1.0
        last_exception = None

        while attempt < self.retries:
            try:
                logger.info(f"Fetching data for {symbol} ({ticker}) from {start_date} to {end_date} (Attempt {attempt+1})")
                df = yf.download(ticker, start=start_date, end=end_date, progress=False, timeout=self.timeout)
                if df is None or df.empty:
                    raise ValueError(f"No data returned for ticker {ticker}")

                # Format multi-index columns if returned by yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df = df.reset_index()

                # Standardize column names
                rename_map = {
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Adj Close": "adj_close",
                    "Volume": "volume"
                }
                df = df.rename(columns=rename_map)

                if "date" not in df.columns:
                    raise KeyError(f"'date' column missing after processing yfinance data for {ticker}")

                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                df["symbol"] = symbol
                df["provider"] = "yfinance"
                df["source"] = ticker
                df["market"] = instrument_meta.get("market", "unknown")
                df["currency"] = instrument_meta.get("currency", "USD")
                df["unit"] = instrument_meta.get("unit", "unknown")
                df["timezone"] = "UTC"
                df["retrieval_timestamp"] = pd.Timestamp.now(tz="UTC").isoformat()

                required_cols = ["date", "open", "high", "low", "close", "volume", "symbol", "provider", "source", "market", "currency", "unit", "timezone", "retrieval_timestamp"]
                for col in ["open", "high", "low", "close", "volume"]:
                    if col not in df.columns:
                        df[col] = 0.0

                return df[required_cols]

            except Exception as e:
                last_exception = e
                attempt += 1
                logger.warning(f"Error fetching {symbol} ({ticker}): {e}. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= self.backoff_factor

        logger.error(f"Failed to fetch data for {symbol} ({ticker}) after {self.retries} attempts.")
        raise RuntimeError(f"Failed to fetch data for {symbol} ({ticker}): {last_exception}")

    def get_economic_events(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Returns macro calendar events."""
        # Baseline economic events calendar structured dataframe
        events = [
            {"date": "2024-01-31", "event": "FOMC_DECISION", "importance": "HIGH", "actual": 5.25, "forecast": 5.25},
            {"date": "2024-03-20", "event": "FOMC_DECISION", "importance": "HIGH", "actual": 5.25, "forecast": 5.25},
            {"date": "2024-05-01", "event": "FOMC_DECISION", "importance": "HIGH", "actual": 5.25, "forecast": 5.25},
            {"date": "2024-06-12", "event": "FOMC_DECISION", "importance": "HIGH", "actual": 5.25, "forecast": 5.25},
            {"date": "2024-09-18", "event": "FOMC_DECISION", "importance": "HIGH", "actual": 4.75, "forecast": 5.00},
            {"date": "2024-11-07", "event": "FOMC_DECISION", "importance": "HIGH", "actual": 4.50, "forecast": 4.50},
            {"date": "2024-12-18", "event": "FOMC_DECISION", "importance": "HIGH", "actual": 4.25, "forecast": 4.25},
            {"date": "2024-02-08", "event": "RBI_DECISION", "importance": "HIGH", "actual": 6.50, "forecast": 6.50},
            {"date": "2024-04-05", "event": "RBI_DECISION", "importance": "HIGH", "actual": 6.50, "forecast": 6.50},
            {"date": "2024-06-07", "event": "RBI_DECISION", "importance": "HIGH", "actual": 6.50, "forecast": 6.50},
            {"date": "2024-08-08", "event": "RBI_DECISION", "importance": "HIGH", "actual": 6.50, "forecast": 6.50},
            {"date": "2024-10-09", "event": "RBI_DECISION", "importance": "HIGH", "actual": 6.50, "forecast": 6.50},
            {"date": "2024-12-06", "event": "RBI_DECISION", "importance": "HIGH", "actual": 6.50, "forecast": 6.50},
        ]
        df = pd.DataFrame(events)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        return df


class MockMarketDataProvider(MarketDataProvider):
    """Deterministic mock provider for testing and fallback."""

    def __init__(self, df_dict: Optional[Dict[str, pd.DataFrame]] = None):
        self.df_dict = df_dict or {}

    def get_historical_prices(
        self,
        symbol: str,
        ticker: str,
        start_date: str,
        end_date: str,
        instrument_meta: Dict[str, Any]
    ) -> pd.DataFrame:
        if symbol in self.df_dict:
            df = self.df_dict[symbol].copy()
            df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
            return df

        # Generate synthetic realistic data for testing
        dates = pd.date_range(start=start_date, end=end_date, freq="B").strftime("%Y-%m-%d")
        import numpy as np
        np.random.seed(42 + hash(symbol) % 1000)
        n = len(dates)
        if n == 0:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "symbol", "provider", "source", "market", "currency", "unit", "timezone", "retrieval_timestamp"])

        base_price = 2000.0 if "GOLD" in symbol else (25.0 if "SILVER" in symbol else (83.0 if "USDINR" in symbol else 100.0))
        returns = np.random.normal(0.0003, 0.01, n)
        price_path = base_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            "date": dates,
            "open": price_path * 0.998,
            "high": price_path * 1.005,
            "low": price_path * 0.995,
            "close": price_path,
            "volume": np.random.randint(1000, 50000, n),
            "symbol": symbol,
            "provider": "mock",
            "source": ticker,
            "market": instrument_meta.get("market", "spot"),
            "currency": instrument_meta.get("currency", "USD"),
            "unit": instrument_meta.get("unit", "oz"),
            "timezone": "UTC",
            "retrieval_timestamp": pd.Timestamp.now(tz="UTC").isoformat()
        })
        return df

    def get_economic_events(self, start_date: str, end_date: str) -> pd.DataFrame:
        events = [
            {"date": start_date, "event": "FOMC_DECISION", "importance": "HIGH", "actual": 5.25, "forecast": 5.25},
            {"date": start_date, "event": "CPI_RELEASE", "importance": "HIGH", "actual": 3.1, "forecast": 3.1}
        ]
        df = pd.DataFrame(events)
        return df
