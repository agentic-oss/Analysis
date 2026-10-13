"""Base provider interface for market and macro data ingestion."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd


class MarketDataProvider(ABC):
    """Abstract base class for precious metals and macro data ingestion."""

    @abstractmethod
    def get_spot_prices(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch spot price series."""
        pass

    @abstractmethod
    def get_historical_prices(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical daily OHLCV dataframe."""
        pass

    @abstractmethod
    def get_futures_prices(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch futures price series."""
        pass

    @abstractmethod
    def get_currency_prices(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch currency exchange rate series."""
        pass

    @abstractmethod
    def get_macro_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch macroeconomic series (e.g. DXY, yields)."""
        pass

    @abstractmethod
    def get_economic_indicators(self, indicator: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch economic indicator series."""
        pass

    @abstractmethod
    def get_event_data(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Fetch central bank and economic events."""
        pass
