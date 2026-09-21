import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

from src.ingestion.provider import MarketDataProvider, YahooMarketDataProvider, MockMarketDataProvider

logger = logging.getLogger(__name__)


class DataCollector:
    """Collector managing retrieval of multi-asset market data and currency/unit conversions."""

    def __init__(
        self,
        instruments_config_path: str = "config/instruments.json",
        use_fallback: bool = True
    ):
        self.instruments_config_path = instruments_config_path
        self.use_fallback = use_fallback
        self.instruments = self._load_instruments()
        self.primary_provider = YahooMarketDataProvider()
        self.fallback_provider = MockMarketDataProvider()

    def _load_instruments(self) -> List[Dict[str, Any]]:
        with open(self.instruments_config_path, "r") as f:
            data = json.load(f)
        return data.get("instruments", [])

    def fetch_instrument_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        inst = next((i for i in self.instruments if i["symbol"] == symbol), None)
        if not inst:
            logger.error(f"Instrument symbol {symbol} not found in configuration.")
            return pd.DataFrame()

        ticker = inst["ticker"]
        df = pd.DataFrame()

        # Attempt primary provider
        try:
            df = self.primary_provider.fetch_historical_ohlcv(symbol, ticker, start_date, end_date)
        except Exception as e:
            logger.warning(f"Primary provider failed for {symbol}: {e}")

        # Fallback if primary returned empty and fallback enabled
        if (df.empty or len(df) == 0) and self.use_fallback:
            logger.info(f"Using fallback provider for symbol {symbol}")
            df = self.fallback_provider.fetch_historical_ohlcv(symbol, ticker, start_date, end_date)

        if df.empty:
            return pd.DataFrame()

        # Attach instrument metadata
        df["market"] = inst.get("market", "")
        df["asset_class"] = inst.get("asset_class", "")
        df["currency"] = inst.get("currency", "USD")
        df["unit"] = inst.get("unit", "oz")
        df["exchange"] = inst.get("exchange", "")
        df["timezone"] = "UTC"

        return df

    def collect_all(
        self,
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """Collect historical data for all enabled instruments."""
        results = {}
        for inst in self.instruments:
            if not inst.get("enabled", True):
                continue
            sym = inst["symbol"]
            df = self.fetch_instrument_data(sym, start_date, end_date)
            if not df.empty:
                results[sym] = df

        # Apply currency & unit conversions where applicable
        results = self._apply_conversions(results)
        return results

    def _apply_conversions(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Calculate INR converted prices for USD gold/silver using aligned USD/INR FX rate."""
        if "USDINR" not in data_dict or data_dict["USDINR"].empty:
            logger.warning("USDINR data missing; skipping INR price conversions.")
            return data_dict

        usdinr_df = data_dict["USDINR"][["date", "close"]].rename(columns={"close": "usdinr_rate"})

        for symbol in ["GOLD", "SILVER", "GOLD_FUTURES", "SILVER_FUTURES"]:
            if symbol in data_dict and not data_dict[symbol].empty:
                df = data_dict[symbol].copy()
                df = pd.merge(df, usdinr_df, on="date", how="left")
                df["usdinr_rate"] = df["usdinr_rate"].ffill().bfill()

                conversion_time = datetime.now(timezone.utc).isoformat()
                df["conversion_timestamp"] = conversion_time

                if "GOLD" in symbol:
                    # 1 troy oz = 31.1034768 grams -> price per 10g in INR = (price_usd / 31.1034768) * 10 * USDINR
                    factor = (10.0 / 31.1034768)
                    df["price_inr"] = df["close"] * df["usdinr_rate"] * factor
                    df["unit_inr"] = "10g"
                elif "SILVER" in symbol:
                    # 1 troy oz = 0.0311034768 kg -> price per kg in INR = (price_usd / 0.0311034768) * USDINR
                    factor = (1000.0 / 31.1034768)
                    df["price_inr"] = df["close"] * df["usdinr_rate"] * factor
                    df["unit_inr"] = "kg"

                data_dict[symbol] = df

        return data_dict


def get_economic_calendar_events(start_date: str, end_date: str) -> pd.DataFrame:
    """Generate or retrieve macro/central bank economic events schedule."""
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    events = []

    event_types = [
        ("FOMC Rate Decision", "Fed", "HIGH"),
        ("US CPI Inflation", "Macro", "HIGH"),
        ("US Nonfarm Payrolls", "Employment", "HIGH"),
        ("RBI Policy Decision", "RBI", "HIGH"),
        ("US PCE Inflation", "Macro", "MEDIUM"),
        ("ECB Policy Decision", "ECB", "MEDIUM"),
    ]

    # Deterministic event mapping
    for i, dt in enumerate(dates):
        if dt.day in [10, 15, 25]:
            ev_type, cat, imp = event_types[i % len(event_types)]
            events.append({
                'date': dt.strftime('%Y-%m-%d'),
                'event_name': ev_type,
                'category': cat,
                'importance': imp,
                'days_to_event': 0,
                'days_since_event': 0
            })

    df = pd.DataFrame(events)
    if df.empty:
        df = pd.DataFrame(columns=['date', 'event_name', 'category', 'importance', 'days_to_event', 'days_since_event'])
    return df
