import abc
from typing import Dict, Any, Optional
import pandas as pd

class MarketDataProvider(abc.ABC):
    """Abstract base class for precious metals and market data providers."""

    @abc.abstractmethod
    def get_historical_prices(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        ticker_override: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical price data for a given instrument symbol.
        Returns a DataFrame with standard columns:
        [date, open, high, low, close, volume, provider, source, retrieval_timestamp, instrument, market, currency, unit, timezone]
        """
        pass

    @abc.abstractmethod
    def get_economic_indicators(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch economic events and macro indicators."""
        pass
