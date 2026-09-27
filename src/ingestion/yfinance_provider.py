import time
import logging
import json
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Any, Optional, List
from src.ingestion.provider import MarketDataProvider

logger = logging.getLogger(__name__)

class YFinanceProvider(MarketDataProvider):
    """Data provider using yfinance with exponential backoff and retries."""

    def __init__(self, config_path: str = "config/instruments.json", max_retries: int = 3, retry_delay: float = 2.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        with open(config_path, "r") as f:
            self.instruments_meta = json.load(f)["instruments"]
        self.instruments_map = {inst["symbol"]: inst for inst in self.instruments_meta}

    def _fetch_with_retry(self, ticker: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        delay = self.retry_delay
        for attempt in range(self.max_retries):
            try:
                t = yf.Ticker(ticker)
                df = t.history(start=start, end=end, auto_adjust=False)
                if df is not None and not df.empty:
                    return df
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed for ticker {ticker}: {e}")
            time.sleep(delay)
            delay *= 2.0
        return pd.DataFrame()

    def get_historical_prices(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        ticker_override: Optional[str] = None
    ) -> pd.DataFrame:
        meta = self.instruments_map.get(symbol, {})
        ticker = ticker_override or meta.get("ticker", symbol)

        df = self._fetch_with_retry(ticker, start=start_date, end=end_date)
        if df.empty:
            logger.error(f"Failed to retrieve data for symbol {symbol} (ticker {ticker})")
            return pd.DataFrame()

        df = df.reset_index()
        # standardise columns
        col_map = {col: col.lower() for col in df.columns}
        df = df.rename(columns=col_map)

        if "date" not in df.columns and "datetime" in df.columns:
            df["date"] = df["datetime"]

        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d")

        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                df[col] = np.nan

        out = pd.DataFrame()
        out["date"] = df["date"]
        out["open"] = df["open"]
        out["high"] = df["high"]
        out["low"] = df["low"]
        out["close"] = df["close"]
        out["volume"] = df["volume"]
        out["provider"] = "yfinance"
        out["source"] = ticker
        out["retrieval_timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        out["instrument"] = symbol
        out["market"] = meta.get("market", "unknown")
        out["currency"] = meta.get("currency", "USD")
        out["unit"] = meta.get("unit", "units")
        out["timezone"] = "UTC"

        out = out.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
        return out

    def get_economic_indicators(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        # Structured event records for CPI, FOMC, RBI, PCE
        events = [
            {"date": "2024-01-31", "event": "FOMC_DECISION", "importance": "HIGH", "category": "central_bank", "actual": "5.50", "forecast": "5.50"},
            {"date": "2024-02-13", "event": "US_CPI", "importance": "HIGH", "category": "inflation", "actual": "3.1", "forecast": "2.9"},
            {"date": "2024-03-20", "event": "FOMC_DECISION", "importance": "HIGH", "category": "central_bank", "actual": "5.50", "forecast": "5.50"},
            {"date": "2024-04-10", "event": "US_CPI", "importance": "HIGH", "category": "inflation", "actual": "3.5", "forecast": "3.4"},
            {"date": "2024-05-01", "event": "FOMC_DECISION", "importance": "HIGH", "category": "central_bank", "actual": "5.50", "forecast": "5.50"},
            {"date": "2024-06-12", "event": "FOMC_DECISION", "importance": "HIGH", "category": "central_bank", "actual": "5.50", "forecast": "5.50"},
            {"date": "2024-07-31", "event": "FOMC_DECISION", "importance": "HIGH", "category": "central_bank", "actual": "5.50", "forecast": "5.50"},
            {"date": "2024-09-18", "event": "FOMC_DECISION", "importance": "HIGH", "category": "central_bank", "actual": "5.00", "forecast": "5.25"},
            {"date": "2024-11-07", "event": "FOMC_DECISION", "importance": "HIGH", "category": "central_bank", "actual": "4.75", "forecast": "4.75"},
            {"date": "2024-12-18", "event": "FOMC_DECISION", "importance": "HIGH", "category": "central_bank", "actual": "4.50", "forecast": "4.50"},
            {"date": "2025-01-29", "event": "FOMC_DECISION", "importance": "HIGH", "category": "central_bank", "actual": "4.50", "forecast": "4.50"},
            {"date": "2025-03-19", "event": "FOMC_DECISION", "importance": "HIGH", "category": "central_bank", "actual": "4.50", "forecast": "4.50"}
        ]
        df = pd.DataFrame(events)
        if start_date:
            df = df[df["date"] >= start_date]
        if end_date:
            df = df[df["date"] <= end_date]
        return df.reset_index(drop=True)
