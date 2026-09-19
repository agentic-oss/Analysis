import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import pandas as pd

logger = logging.getLogger(__name__)

class MarketDataProvider(ABC):
    """Abstract Base Class for Market Data Providers."""

    @abstractmethod
    def fetch_historical_prices(
        self,
        symbol: str,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch historical price data (OHLCV) for an instrument."""
        pass

    @abstractmethod
    def fetch_latest_quote(self, symbol: str, ticker: str) -> Dict[str, Any]:
        """Fetch latest quote / daily bar for an instrument."""
        pass


class YFinanceProvider(MarketDataProvider):
    """yfinance market data provider implementation with retries and rate limit handling."""

    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0, timeout: int = 15):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.timeout = timeout

    def _execute_with_retry(self, func, *args, **kwargs):
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                sleep_time = self.backoff_factor ** attempt
                logger.warning(f"Data fetch attempt {attempt + 1} failed: {e}. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
        logger.error(f"Failed after {self.max_retries} attempts.")
        raise last_exception

    def fetch_historical_prices(
        self,
        symbol: str,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        import yfinance as yf

        def _download():
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, timeout=self.timeout)
            if df.empty:
                return pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            df = df.reset_index()
            # Standardize columns
            col_map = {col: str(col).lower() for col in df.columns}
            df = df.rename(columns=col_map)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            df['symbol'] = symbol
            df['ticker'] = ticker
            df['provider'] = 'yfinance'
            df['retrieved_at'] = datetime.now(timezone.utc).isoformat()
            return df

        try:
            return self._execute_with_retry(_download)
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol} ({ticker}): {e}")
            return pd.DataFrame()

    def fetch_latest_quote(self, symbol: str, ticker: str) -> Dict[str, Any]:
        import yfinance as yf

        def _get_quote():
            t = yf.Ticker(ticker)
            hist = t.history(period="5d", timeout=self.timeout)
            if hist.empty:
                return {}
            latest = hist.iloc[-1]
            return {
                "symbol": symbol,
                "ticker": ticker,
                "date": hist.index[-1].strftime('%Y-%m-%d'),
                "open": float(latest.get("Open", 0.0)),
                "high": float(latest.get("High", 0.0)),
                "low": float(latest.get("Low", 0.0)),
                "close": float(latest.get("Close", 0.0)),
                "volume": float(latest.get("Volume", 0.0)),
                "provider": "yfinance",
                "retrieved_at": datetime.now(timezone.utc).isoformat()
            }

        try:
            return self._execute_with_retry(_get_quote)
        except Exception as e:
            logger.error(f"Error fetching latest quote for {symbol} ({ticker}): {e}")
            return {}


class MockMarketDataProvider(MarketDataProvider):
    """Deterministic mock provider for testing and offline data generation."""

    def __init__(self, df_dict: Optional[Dict[str, pd.DataFrame]] = None):
        self.df_dict = df_dict or {}

    def fetch_historical_prices(
        self,
        symbol: str,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        if symbol in self.df_dict:
            df = self.df_dict[symbol].copy()
            if 'date' in df.columns:
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            return df
        return pd.DataFrame()

    def fetch_latest_quote(self, symbol: str, ticker: str) -> Dict[str, Any]:
        if symbol in self.df_dict and not self.df_dict[symbol].empty:
            df = self.df_dict[symbol]
            row = df.iloc[-1]
            return {
                "symbol": symbol,
                "ticker": ticker,
                "date": str(row.get("date", "2026-03-31")),
                "open": float(row.get("open", 100.0)),
                "high": float(row.get("high", 105.0)),
                "low": float(row.get("low", 95.0)),
                "close": float(row.get("close", 102.0)),
                "volume": float(row.get("volume", 1000.0)),
                "provider": "mock",
                "retrieved_at": datetime.now(timezone.utc).isoformat()
            }
        return {}
