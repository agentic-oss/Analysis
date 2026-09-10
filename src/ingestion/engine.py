import json
import logging
import pandas as pd
from typing import Dict, Any, List, Optional
from src.ingestion.provider import MarketDataProvider, YFinanceMarketDataProvider, MockMarketDataProvider

logger = logging.getLogger(__name__)

class DataIngestionEngine:
    """Coordinates fetching and normalizing data from MarketDataProvider for all configured instruments."""

    def __init__(self, config_path: str = "config/config.yaml", instruments_path: str = "config/instruments.json", provider: Optional[MarketDataProvider] = None):
        self.instruments_path = instruments_path
        with open(instruments_path, "r") as f:
            self.instruments_config = json.load(f)

        self.provider = provider or YFinanceMarketDataProvider()

    def fetch_all_instruments(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        data = {}
        instruments = self.instruments_config.get("instruments", [])

        for inst in instruments:
            if not inst.get("enabled", True):
                continue
            symbol = inst["symbol"]
            ticker = inst.get("provider_ticker", symbol)
            try:
                df = self.provider.get_historical_prices(symbol, ticker, start_date, end_date, inst)
                data[symbol] = df
            except Exception as e:
                logger.error(f"Error ingesting {symbol}: {e}")

        return data

    @staticmethod
    def calculate_inr_converted_prices(
        gold_df: pd.DataFrame,
        silver_df: pd.DataFrame,
        usdinr_df: pd.DataFrame,
        oz_to_grams: float = 31.1034768
    ) -> Dict[str, pd.DataFrame]:
        """Convert Spot Gold and Spot Silver prices in USD/oz to INR/10g and INR/kg using exact USDINR rates."""
        converted = {}

        if usdinr_df.empty:
            return converted

        usdinr_map = usdinr_df.set_index("date")["close"].to_dict()

        if not gold_df.empty:
            df_g = gold_df.copy()
            df_g["usdinr"] = df_g["date"].map(usdinr_map)
            # Price in USD/oz -> INR per 10 grams = (price / 31.1034768) * usdinr * 10
            df_g["converted_inr_10g"] = (df_g["close"] / oz_to_grams) * df_g["usdinr"] * 10.0
            df_g["conversion_rate_used"] = df_g["usdinr"]
            df_g["conversion_timestamp"] = pd.Timestamp.now(tz="UTC").isoformat()
            converted["GOLD_DERIVED_INR"] = df_g

        if not silver_df.empty:
            df_s = silver_df.copy()
            df_s["usdinr"] = df_s["date"].map(usdinr_map)
            # Price in USD/oz -> INR per kg = (price / 31.1034768) * usdinr * 1000
            df_s["converted_inr_kg"] = (df_s["close"] / oz_to_grams) * df_s["usdinr"] * 1000.0
            df_s["conversion_rate_used"] = df_s["usdinr"]
            df_s["conversion_timestamp"] = pd.Timestamp.now(tz="UTC").isoformat()
            converted["SILVER_DERIVED_INR"] = df_s

        return converted
