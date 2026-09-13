import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd

logger = logging.getLogger(__name__)

class MarketDataProvider(ABC):
    """Abstract Base Class for Market Data Providers."""

    @abstractmethod
    def fetch_historical(
        self,
        symbol: str,
        ticker: str,
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical market OHLCV data.
        Returns DataFrame indexed by Datetime/Date with columns:
        [Open, High, Low, Close, Volume, provider, symbol, market, currency, unit, timezone, retrieval_timestamp]
        """
        pass

    @abstractmethod
    def fetch_latest(self, symbol: str, ticker: str) -> Dict[str, Any]:
        """Fetch latest single quote."""
        pass


def retry_with_backoff(max_retries: int = 3, backoff_factor: float = 1.5):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            delay = 1.0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        logger.error(f"Failed after {retries} retries: {str(e)}")
                        raise e
                    logger.warning(f"Error encountered ({e}). Retrying in {delay:.2f}s... (Attempt {retries}/{max_retries})")
                    time.sleep(delay)
                    delay *= backoff_factor
        return wrapper
    return decorator
