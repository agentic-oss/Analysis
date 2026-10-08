"""
MarketDataProvider Interface definition.
"""
import abc
import datetime
from typing import Dict, List, Optional, Any
import pandas as pd


class MarketDataProvider(abc.ABC):
    """
    Abstract Base Class for precious metals and market data providers.
    """

    @abc.abstractmethod
    def get_historical_prices(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical daily OHLCV prices for a given instrument symbol.
        DataFrame must contain standardized columns:
        ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'currency', 'unit', 'provider', 'retrieval_timestamp']
        """
        pass

    @abc.abstractmethod
    def get_spot_prices(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetch latest spot/current price snapshot for a list of instrument symbols.
        """
        pass

    @abc.abstractmethod
    def fetch_all_instruments_data(
        self,
        instruments_config: List[Dict[str, Any]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all enabled instruments in config.
        """
        pass
