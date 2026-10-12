import time
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

TROY_OUNCE_TO_GRAMS = 31.1034768

class MarketDataProvider(ABC):
    """Abstract Base Class for Market Data Providers."""

    @abstractmethod
    def fetch_historical_series(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch historical price series for a symbol."""
        pass

    @abstractmethod
    def get_spot_prices(self, symbols: List[str]) -> Dict[str, Any]:
        """Get latest spot prices for given symbols."""
        pass


class YahooMarketDataProvider(MarketDataProvider):
    """Data provider using yfinance with exponential backoff and error handling."""

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def fetch_historical_series(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        import yfinance as yf

        retries = 0
        delay = self.retry_delay
        while retries <= self.max_retries:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval=interval)
                if df.empty:
                    logger.warning(f"Empty data returned for symbol {symbol}")
                    return pd.DataFrame()

                df = df.reset_index()
                # Ensure standard columns
                rename_cols = {
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume"
                }
                df = df.rename(columns=rename_cols)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

                df["symbol"] = symbol
                df["provider"] = "yahoo"
                df["retrieval_timestamp"] = datetime.now(timezone.utc).isoformat()
                return df
            except Exception as e:
                retries += 1
                if retries > self.max_retries:
                    logger.error(f"Failed to fetch data for {symbol} after {self.max_retries} retries: {e}")
                    raise
                time.sleep(delay)
                delay *= 2

    def get_spot_prices(self, symbols: List[str]) -> Dict[str, Any]:
        results = {}
        for sym in symbols:
            df = self.fetch_historical_series(sym, start_date=(datetime.now().strftime("%Y-%m-%d")), end_date=(datetime.now().strftime("%Y-%m-%d")))
            if not df.empty:
                results[sym] = df.iloc[-1].to_dict()
        return results


class SyntheticMarketDataProvider(MarketDataProvider):
    """Deterministic synthetic data generator for offline testing, backfilling, and CI."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fetch_historical_series(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        np.random.seed(self.seed + abs(hash(symbol)) % 10000)
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        if len(dates) == 0:
            return pd.DataFrame()

        # Base prices for different symbols
        base_prices = {
            "GC=F": 2000.0,
            "SI=F": 24.0,
            "INR=X": 83.0,
            "DX-Y.NYB": 103.0,
            "^TNX": 4.2,
            "^IRX": 5.0,
            "TIP": 108.0,
            "CL=F": 75.0,
            "BZ=F": 80.0,
            "^GSPC": 5000.0,
            "^IXIC": 16000.0,
            "^DJI": 38000.0,
            "^NSEI": 22000.0,
            "^VIX": 15.0,
            "^INDIAVIX": 14.0,
            "GOLDBEES.NS": 60.0,
            "SILVERBEES.NS": 75.0
        }
        base_price = base_prices.get(symbol, 100.0)

        # Volatilities
        vol = 0.012 if "GC=F" in symbol or "SI=F" in symbol else 0.01

        returns = np.random.normal(0.0003, vol, len(dates))
        price_paths = base_price * np.exp(np.cumsum(returns))

        records = []
        for d, p in zip(dates, price_paths):
            high = p * (1 + abs(np.random.normal(0, 0.005)))
            low = p * (1 - abs(np.random.normal(0, 0.005)))
            open_p = low + (high - low) * np.random.uniform(0.2, 0.8)
            close_p = low + (high - low) * np.random.uniform(0.2, 0.8)
            vol_val = int(np.random.uniform(1000, 50000))

            records.append({
                "date": d.strftime("%Y-%m-%d"),
                "symbol": symbol,
                "open": round(float(open_p), 4),
                "high": round(float(high), 4),
                "low": round(float(low), 4),
                "close": round(float(close_p), 4),
                "volume": vol_val,
                "provider": "synthetic",
                "retrieval_timestamp": datetime.now(timezone.utc).isoformat()
            })

        return pd.DataFrame(records)

    def get_spot_prices(self, symbols: List[str]) -> Dict[str, Any]:
        results = {}
        for sym in symbols:
            df = self.fetch_historical_series(sym, start_date="2026-01-01", end_date="2026-01-05")
            if not df.empty:
                results[sym] = df.iloc[-1].to_dict()
        return results


def convert_usd_to_inr_metrics(
    gold_usd_oz: float,
    silver_usd_oz: float,
    usdinr_rate: float
) -> Dict[str, float]:
    """Convert USD/oz prices to Indian standard units (INR/10g for gold, INR/kg for silver)."""
    gold_inr_per_10g = (gold_usd_oz * usdinr_rate / TROY_OUNCE_TO_GRAMS) * 10.0
    silver_inr_per_kg = (silver_usd_oz * usdinr_rate / TROY_OUNCE_TO_GRAMS) * 1000.0
    return {
        "gold_inr_per_10g": round(gold_inr_per_10g, 2),
        "silver_inr_per_kg": round(silver_inr_per_kg, 2),
        "usdinr_rate": usdinr_rate
    }
