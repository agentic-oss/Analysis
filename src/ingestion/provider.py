import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class MarketDataProvider(ABC):
    """Abstract Base Class for Precious Metals & Macro Data Providers."""

    def __init__(self, name: str, max_retries: int = 3, retry_backoff: float = 2.0):
        self.name = name
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    def execute_with_retry(self, func, *args, **kwargs):
        """Executes a function with retry logic and exponential backoff."""
        attempt = 0
        while attempt < self.max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                attempt += 1
                logger.warning(
                    f"[{self.name}] Attempt {attempt}/{self.max_retries} failed for {func.__name__}: {e}"
                )
                if attempt >= self.max_retries:
                    logger.error(
                        f"[{self.name}] All {self.max_retries} attempts failed for {func.__name__}"
                    )
                    raise e
                time.sleep(self.retry_backoff ** attempt)

    @abstractmethod
    def get_historical_prices(
        self,
        symbol: str,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Retrieves historical OHLCV data for a given symbol/ticker."""
        pass

    @abstractmethod
    def get_spot_prices(self, symbol: str, ticker: str) -> Dict[str, Any]:
        """Retrieves latest spot/quote price."""
        pass

    @abstractmethod
    def get_futures_prices(
        self, symbol: str, ticker: str, start_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Retrieves historical or latest futures price data."""
        pass

    @abstractmethod
    def get_currency_prices(
        self, symbol: str, ticker: str, start_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Retrieves currency exchange rates (e.g. USD/INR)."""
        pass

    @abstractmethod
    def get_macro_data(
        self, symbol: str, ticker: str, start_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Retrieves macroeconomic data (yields, indices, energy)."""
        pass

    @abstractmethod
    def get_economic_indicators(self) -> pd.DataFrame:
        """Retrieves calendar / economic indicators (CPI, Central Bank decisions)."""
        pass
