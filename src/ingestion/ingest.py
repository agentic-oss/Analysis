"""
Data Ingestion Service.
Orchestrates fetching raw data from MarketDataProvider, handling multi-market instruments and fallback logic.
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
from src.ingestion.provider import MarketDataProvider, YFinanceProvider, SyntheticProvider

logger = logging.getLogger(__name__)


class DataIngestionService:
    def __init__(self, primary_provider: MarketDataProvider, fallback_provider: Optional[MarketDataProvider] = None):
        self.primary_provider = primary_provider
        self.fallback_provider = fallback_provider or SyntheticProvider()

    def fetch_instrument_history(
        self,
        instrument: Dict[str, Any],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch historical data for a given instrument definition with fallback handling."""
        symbol = instrument["symbol"]
        ticker = instrument.get("ticker", symbol)

        try:
            df = self.primary_provider.fetch_historical_data(
                symbol=symbol,
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
            )
            df["currency"] = instrument.get("currency", "USD")
            df["unit"] = instrument.get("unit", "oz")
            df["market"] = instrument.get("market", "spot")
            return df
        except Exception as e:
            logger.warning(f"Primary provider failed for {symbol}: {e}. Falling back to fallback provider.")
            df = self.fallback_provider.fetch_historical_data(
                symbol=symbol,
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
            )
            df["currency"] = instrument.get("currency", "USD")
            df["unit"] = instrument.get("unit", "oz")
            df["market"] = instrument.get("market", "spot")
            return df
