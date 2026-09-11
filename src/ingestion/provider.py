"""
Market Data Provider abstraction and implementations.
Supports resilient data retrieval, retries, rate-limiting, and synthetic fallback.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import logging
import time
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class MarketDataProvider(ABC):
    """Abstract Base Class for Market Data Providers."""

    @abstractmethod
    def fetch_historical_data(
        self,
        symbol: str,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical price/OHLCV data for a given ticker."""
        pass

    @abstractmethod
    def fetch_latest_quote(self, symbol: str, ticker: str) -> Dict[str, Any]:
        """Fetch latest single quote."""
        pass


class YFinanceProvider(MarketDataProvider):
    """Yahoo Finance Market Data Provider implementation with retries and backoff."""

    def __init__(self, retries: int = 3, backoff_sec: float = 2.0, timeout: int = 15):
        self.retries = retries
        self.backoff_sec = backoff_sec
        self.timeout = timeout

    def fetch_historical_data(
        self,
        symbol: str,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical data from yfinance with retry logic."""
        last_exception = None
        for attempt in range(1, self.retries + 1):
            try:
                logger.info(f"Fetching {symbol} ({ticker}) attempt {attempt}/{self.retries}...")
                df = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    progress=False,
                    auto_adjust=False,
                    timeout=self.timeout,
                )
                if df.empty:
                    raise ValueError(f"No data returned for ticker {ticker}")

                # Clean multi-index columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df = df.reset_index()
                # Normalize column names
                rename_dict = {
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Adj Close": "adj_close",
                    "Volume": "volume",
                }
                df = df.rename(columns=rename_dict)

                # Ensure date column is formatted YYYY-MM-DD string
                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

                # Fill adj_close if missing
                if "adj_close" not in df.columns:
                    df["adj_close"] = df["close"]

                df["symbol"] = symbol
                df["provider"] = "yfinance"
                df["retrieved_at"] = datetime.now(timezone.utc).isoformat()
                return df
            except Exception as e:
                last_exception = e
                logger.warning(f"Error fetching {symbol} ({ticker}): {e}. Retrying in {self.backoff_sec * attempt}s...")
                time.sleep(self.backoff_sec * attempt)

        logger.error(f"Failed to fetch {symbol} ({ticker}) after {self.retries} attempts.")
        raise RuntimeError(f"Failed to fetch {symbol} ({ticker}): {last_exception}")

    def fetch_latest_quote(self, symbol: str, ticker: str) -> Dict[str, Any]:
        """Fetch latest quote from yfinance."""
        ticker_obj = yf.Ticker(ticker)
        hist = ticker_obj.history(period="5d")
        if hist.empty:
            raise ValueError(f"No quote data available for {ticker}")
        latest = hist.iloc[-1]
        return {
            "symbol": symbol,
            "ticker": ticker,
            "date": latest.name.strftime("%Y-%m-%d"),
            "open": float(latest["Open"]),
            "high": float(latest["High"]),
            "low": float(latest["Low"]),
            "close": float(latest["Close"]),
            "volume": float(latest["Volume"]) if "Volume" in latest else 0.0,
            "provider": "yfinance",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }


class SyntheticProvider(MarketDataProvider):
    """Deterministic Synthetic Market Data Provider for offline testing, verification, and fallback."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fetch_historical_data(
        self,
        symbol: str,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        np.random.seed(self.seed + sum(ord(c) for c in symbol))

        n = len(dates)
        if n == 0:
            return pd.DataFrame()

        # Base prices depending on symbol
        base_prices = {
            "GOLD": 2000.0,
            "SILVER": 25.0,
            "GOLD_FUTURES": 2010.0,
            "SILVER_FUTURES": 25.2,
            "USDINR": 83.0,
            "DXY": 104.0,
            "US10Y": 4.2,
            "US2Y": 4.5,
            "REAL_YIELD": 1.8,
            "CRUDE_OIL": 75.0,
            "SP500": 5000.0,
            "NIFTY50": 22000.0,
            "VIX": 15.0,
            "INDIA_VIX": 13.0,
            "MCX_GOLD": 65000.0,
            "MCX_SILVER": 74000.0,
        }
        base = base_prices.get(symbol, 100.0)
        daily_returns = np.random.normal(0.0002, 0.012, n)
        price_path = base * np.cumprod(1 + daily_returns)

        records = []
        for i, dt in enumerate(dates):
            close_p = float(price_path[i])
            open_p = close_p * (1 + np.random.uniform(-0.005, 0.005))
            high_p = max(open_p, close_p) * (1 + abs(np.random.uniform(0.000, 0.01)))
            low_p = min(open_p, close_p) * (1 - abs(np.random.uniform(0.000, 0.01)))
            volume = float(np.random.randint(10000, 500000))

            records.append({
                "date": dt.strftime("%Y-%m-%d"),
                "symbol": symbol,
                "open": round(open_p, 4),
                "high": round(high_p, 4),
                "low": round(low_p, 4),
                "close": round(close_p, 4),
                "adj_close": round(close_p, 4),
                "volume": volume,
                "provider": "synthetic",
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            })

        return pd.DataFrame(records)

    def fetch_latest_quote(self, symbol: str, ticker: str) -> Dict[str, Any]:
        df = self.fetch_historical_data(symbol, ticker, "2026-01-01", "2026-03-31")
        latest = df.iloc[-1]
        return latest.to_dict()
