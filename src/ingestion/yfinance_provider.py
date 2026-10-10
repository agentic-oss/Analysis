import time
import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import yfinance as yf
from src.ingestion.provider import MarketDataProvider

logger = logging.getLogger(__name__)

class YFinanceProvider(MarketDataProvider):
    """
    MarketDataProvider implementation backed by yfinance with retry and exponential backoff logic.
    """

    def __init__(self, retry_count: int = 3, timeout: int = 15, backoff_factor: float = 2.0):
        self.retry_count = retry_count
        self.timeout = timeout
        self.backoff_factor = backoff_factor

    def _download_with_retry(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        attempt = 0
        backoff = 1.0
        while attempt < self.retry_count:
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False, timeout=self.timeout)
                if not data.empty:
                    # Flatten MultiIndex columns if present (yfinance return format)
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = [col[0] for col in data.columns]
                    data.columns = [str(c).capitalize() for c in data.columns]
                    data.index.name = "Date"
                    return data
                logger.warning(f"Empty data returned for ticker {ticker} on attempt {attempt+1}")
            except Exception as e:
                logger.warning(f"Error fetching {ticker} on attempt {attempt+1}: {str(e)}")

            attempt += 1
            if attempt < self.retry_count:
                time.sleep(backoff)
                backoff *= self.backoff_factor

        logger.error(f"Failed to fetch {ticker} after {self.retry_count} attempts.")
        return pd.DataFrame()

    def get_spot_prices(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self._download_with_retry(symbol, start_date, end_date)

    def get_historical_prices(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self._download_with_retry(symbol, start_date, end_date)

    def get_futures_prices(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self._download_with_retry(symbol, start_date, end_date)

    def get_currency_prices(self, currency_pair: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self._download_with_retry(currency_pair, start_date, end_date)

    def get_macro_data(self, indicator_symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self._download_with_retry(indicator_symbol, start_date, end_date)
