from abc import ABC, abstractmethod
import logging
import time
from typing import Any, Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)

class MarketDataProvider(ABC):
    """Abstract base class for all market data providers."""

    def __init__(self, name: str, retries: int = 3, backoff_factor: float = 2.0, timeout: int = 15):
        self.name = name
        self.retries = retries
        self.backoff_factor = backoff_factor
        self.timeout = timeout

    def _retry_execute(self, func, *args, **kwargs):
        """Executes a callable with exponential backoff retries."""
        last_exception = None
        for attempt in range(1, self.retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"[{self.name}] Attempt {attempt}/{self.retries} failed for {func.__name__}: {e}"
                )
                if attempt < self.retries:
                    time.sleep(self.backoff_factor ** (attempt - 1))
        logger.error(f"[{self.name}] All {self.retries} attempts failed for {func.__name__}.")
        raise last_exception

    @abstractmethod
    def get_spot_prices(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_historical_prices(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_futures_prices(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_currency_prices(self, currency_pair: str, start_date: str, end_date: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_macro_data(self, series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_economic_indicators(self, start_date: str, end_date: str) -> pd.DataFrame:
        pass
