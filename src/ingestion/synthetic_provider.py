import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from src.ingestion.provider import MarketDataProvider

logger = logging.getLogger(__name__)

class SyntheticProvider(MarketDataProvider):
    """
    Deterministic Synthetic MarketDataProvider for testing and fallback offline execution.
    Generates realistic historical price series with realistic trends and correlations.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def _generate_series(self, base_price: float, volatility: float, start_date: str, end_date: str, symbol: str) -> pd.DataFrame:
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        if len(dates) == 0:
            return pd.DataFrame()

        np.random.seed(self.seed + hash(symbol) % 10000)
        returns = np.random.normal(0.0003, volatility, size=len(dates))
        price_paths = base_price * np.exp(np.cumsum(returns))

        high = price_paths * (1 + np.abs(np.random.normal(0, 0.005, size=len(dates))))
        low = price_paths * (1 - np.abs(np.random.normal(0, 0.005, size=len(dates))))
        open_p = low + (high - low) * np.random.uniform(0.2, 0.8, size=len(dates))
        volume = np.random.randint(10000, 500000, size=len(dates))

        df = pd.DataFrame({
            "Open": open_p,
            "High": high,
            "Low": low,
            "Close": price_paths,
            "Adj close": price_paths,
            "Volume": volume
        }, index=dates)
        df.index.name = "Date"
        return df

    def get_spot_prices(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        base = 2000.0 if "GOLD" in symbol.upper() else (25.0 if "SILVER" in symbol.upper() else 100.0)
        vol = 0.012 if "SILVER" in symbol.upper() else 0.008
        return self._generate_series(base, vol, start_date, end_date, symbol)

    def get_historical_prices(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self.get_spot_prices(symbol, start_date, end_date)

    def get_futures_prices(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self.get_spot_prices(symbol, start_date, end_date)

    def get_currency_prices(self, currency_pair: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self._generate_series(83.0 if "INR" in currency_pair else 104.0, 0.004, start_date, end_date, currency_pair)

    def get_macro_data(self, indicator_symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        base = 4.2 if "TNX" in indicator_symbol or "10Y" in indicator_symbol else (18.0 if "VIX" in indicator_symbol else 75.0)
        return self._generate_series(base, 0.015, start_date, end_date, indicator_symbol)
