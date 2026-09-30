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
        self, ticker: str, start_date: str, end_date: str, interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch historical OHLCV market data."""
        pass

    @abstractmethod
    def get_spot_prices(self, tickers: List[str]) -> Dict[str, Any]:
        """Fetch latest snapshot / spot prices."""
        pass


class YahooFinanceProvider(MarketDataProvider):
    """Yahoo Finance implementation with exponential backoff and retries."""

    def __init__(self, max_retries: int = 3, backoff_factor: float = 1.5, timeout: int = 10):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.timeout = timeout

    def get_historical_prices(
        self, ticker: str, start_date: str, end_date: str, interval: str = "1d"
    ) -> pd.DataFrame:
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Fetching {ticker} from {start_date} to {end_date} (attempt {attempt})")
                df = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    progress=False,
                    auto_adjust=False,
                )
                if df.empty:
                    logger.warning(f"No data returned for ticker {ticker}")
                    return pd.DataFrame()

                # Clean multi-index columns if present
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
                    "Volume": "volume",
                }
                df = df.rename(columns=rename_map)
                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                return df
            except Exception as e:
                logger.error(f"Error fetching {ticker} (attempt {attempt}/{self.max_retries}): {e}")
                if attempt == self.max_retries:
                    raise e
                time.sleep(self.backoff_factor ** attempt)
        return pd.DataFrame()

    def get_spot_prices(self, tickers: List[str]) -> Dict[str, Any]:
        results = {}
        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                fast_info = t.fast_info
                results[ticker] = {
                    "last_price": getattr(fast_info, "last_price", None),
                    "previous_close": getattr(fast_info, "previous_close", None),
                    "day_high": getattr(fast_info, "day_high", None),
                    "day_low": getattr(fast_info, "day_low", None),
                    "volume": getattr(fast_info, "last_volume", None),
                }
            except Exception as e:
                logger.error(f"Error fetching spot info for {ticker}: {e}")
                results[ticker] = None
        return results
