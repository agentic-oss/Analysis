from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd

class MarketDataProvider(ABC):
    """
    Abstract Base Class for Market Data Providers.
    """

    @abstractmethod
    def get_spot_prices(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch spot price history."""
        pass

    @abstractmethod
    def get_historical_prices(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch generic historical price series."""
        pass

    @abstractmethod
    def get_futures_prices(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch futures historical data."""
        pass

    @abstractmethod
    def get_currency_prices(self, currency_pair: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch FX currency rates."""
        pass

    @abstractmethod
    def get_macro_data(self, indicator_symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch macroeconomic series."""
        pass
