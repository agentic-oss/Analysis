import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

from src.ingestion.yahoo_provider import YahooMarketDataProvider

logger = logging.getLogger(__name__)


class DataIngestionEngine:
    """Ingestion engine managing collection of precious metals, FX, macro, and Indian market metrics."""

    def __init__(self, instruments_config_path: str = "config/instruments.json"):
        with open(instruments_config_path, "r") as f:
            self.instruments_cfg = json.load(f).get("instruments", [])
        self.provider = YahooMarketDataProvider()

    def fetch_all(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """Fetches historical datasets for all enabled instruments."""
        datasets = {}
        for inst in self.instruments_cfg:
            if not inst.get("enabled", True):
                continue
            symbol = inst["symbol"]
            ticker = inst["ticker"]
            logger.info(f"Ingesting data for {symbol} ({ticker})...")

            try:
                df = self.provider.get_historical_prices(
                    symbol=symbol,
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                )
                if not df.empty:
                    df["market"] = inst.get("market", "")
                    df["currency"] = inst.get("currency", "USD")
                    df["unit"] = inst.get("unit", "")
                    df["category"] = inst.get("category", "")
                    datasets[symbol] = df
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")

        # Post-process currency conversions and Indian market representations
        if "GOLD" in datasets and "USDINR" in datasets:
            datasets["GOLD_INR_CONVERTED"] = self._create_converted_indian_price(
                datasets["GOLD"], datasets["USDINR"], target_symbol="GOLD_INR_CONVERTED", unit="10g"
            )

        if "SILVER" in datasets and "USDINR" in datasets:
            datasets["SILVER_INR_CONVERTED"] = self._create_converted_indian_price(
                datasets["SILVER"], datasets["USDINR"], target_symbol="SILVER_INR_CONVERTED", unit="kg"
            )

        return datasets

    def _create_converted_indian_price(
        self, metal_df: pd.DataFrame, usdinr_df: pd.DataFrame, target_symbol: str, unit: str
    ) -> pd.DataFrame:
        """Explicitly converts USD/oz price to INR/10g or INR/kg storing conversion metadata."""
        merged = pd.merge(
            metal_df[["date", "open", "high", "low", "close", "adj_close", "volume"]],
            usdinr_df[["date", "close"]].rename(columns={"close": "usdinr"}),
            on="date",
            how="inner",
        )
        if merged.empty:
            return pd.DataFrame()

        # Conversion math:
        # 1 troy oz = 31.1034768 grams.
        # For Gold INR/10g: (price_usd / 31.1034768 * 10) * usdinr
        # For Silver INR/kg: (price_usd / 31.1034768 * 1000) * usdinr
        multiplier = (10.0 / 31.1034768) if unit == "10g" else (1000.0 / 31.1034768)

        converted = merged.copy()
        for col in ["open", "high", "low", "close", "adj_close"]:
            converted[col] = converted[col] * converted["usdinr"] * multiplier

        converted["symbol"] = target_symbol
        converted["market"] = "indian_converted"
        converted["currency"] = "INR"
        converted["unit"] = unit
        converted["conversion_rate_used"] = converted["usdinr"]
        converted["conversion_timestamp"] = datetime.utcnow().isoformat()
        converted["provider"] = "calculated"

        return converted.drop(columns=["usdinr"])
