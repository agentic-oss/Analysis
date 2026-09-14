"""
Data Ingestion Module with MarketDataProvider Abstraction Layer.
"""
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketDataProvider(ABC):
    """Abstract Base Class for Precious Metals & Cross-Asset Market Data Providers."""

    @abstractmethod
    def fetch_historical_prices(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
        instrument_meta: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Fetch historical daily OHLCV data for a given symbol.
        Returned DataFrame MUST contain:
        [date, open, high, low, close, volume, symbol, market, currency, unit, provider, source, retrieval_timestamp, timezone]
        """
        pass


class YFinanceMarketDataProvider(MarketDataProvider):
    """
    Yahoo Finance Market Data Provider implementation with retries, exponential backoff, rate limiting, and robust metadata tagger.
    """

    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        timeout: int = 30,
        rate_limit_delay: float = 0.5
    ):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay

    def fetch_historical_prices(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
        instrument_meta: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        time.sleep(self.rate_limit_delay)
        retries = 0
        df = pd.DataFrame()

        while retries <= self.max_retries:
            try:
                logger.info(f"Fetching data for symbol: {symbol} (attempt {retries + 1}/{self.max_retries + 1})")
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval=interval, auto_adjust=False)
                if not df.empty:
                    break
            except Exception as e:
                logger.warning(f"Error fetching {symbol} on attempt {retries + 1}: {e}")

            retries += 1
            if retries <= self.max_retries:
                sleep_time = self.backoff_factor ** retries
                time.sleep(sleep_time)

        if df.empty:
            logger.error(f"Failed to fetch data for symbol: {symbol}")
            return pd.DataFrame()

        df = df.reset_index()
        # Standardize columns
        cols = {col: col.lower().replace(" ", "_") for col in df.columns}
        df = df.rename(columns=cols)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        elif "datetime" in df.columns:
            df["date"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d")

        # Standardize required columns
        for col in ["open", "high", "low", "close", "adj_close", "volume"]:
            if col not in df.columns:
                if col == "adj_close" and "close" in df.columns:
                    df["adj_close"] = df["close"]
                elif col == "volume":
                    df["volume"] = 0
                else:
                    df[col] = np.nan

        # Attach metadata
        meta = instrument_meta or {}
        df["symbol"] = symbol
        df["market"] = meta.get("market", "unknown")
        df["currency"] = meta.get("currency", "USD")
        df["unit"] = meta.get("unit", "unknown")
        df["provider"] = "yfinance"
        df["source"] = f"yfinance:{symbol}"
        df["retrieval_timestamp"] = datetime.now(timezone.utc).isoformat()
        df["timezone"] = "UTC"

        req_cols = [
            "date", "open", "high", "low", "close", "adj_close", "volume",
            "symbol", "market", "currency", "unit", "provider", "source",
            "retrieval_timestamp", "timezone"
        ]
        return df[req_cols]


class SyntheticMarketDataProvider(MarketDataProvider):
    """
    Deterministic Synthetic Market Data Provider for offline testing and fallback simulation.
    Generates realistic historical prices for Gold, Silver, Currencies, Equities, Rates, and Volatility.
    """

    def fetch_historical_prices(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
        instrument_meta: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        if len(dates) == 0:
            return pd.DataFrame()

        np.random.seed(abs(hash(symbol)) % (2**32 - 1))
        n = len(dates)

        # Initial price seed based on symbol
        base_price = 2000.0
        daily_vol = 0.012
        if "SI" in symbol or "SILVER" in symbol:
            base_price = 25.0
            daily_vol = 0.02
        elif "INR" in symbol:
            base_price = 83.0
            daily_vol = 0.003
        elif "DX" in symbol:
            base_price = 104.0
            daily_vol = 0.005
        elif "TNX" in symbol or "IRX" in symbol:
            base_price = 4.2
            daily_vol = 0.015
        elif "CL" in symbol or "BZ" in symbol:
            base_price = 75.0
            daily_vol = 0.02
        elif "VIX" in symbol:
            base_price = 15.0
            daily_vol = 0.03
        elif "GSPC" in symbol or "NSEI" in symbol or "IXIC" in symbol or "DJI" in symbol:
            base_price = 5000.0 if "GSPC" in symbol else (22000.0 if "NSEI" in symbol else 16000.0)
            daily_vol = 0.01

        returns = np.random.normal(0.0002, daily_vol, n)
        price_series = base_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "open": price_series * (1 - 0.003 * np.random.random(n)),
            "high": price_series * (1 + 0.007 * np.random.random(n)),
            "low": price_series * (1 - 0.007 * np.random.random(n)),
            "close": price_series,
            "adj_close": price_series,
            "volume": np.random.randint(1000, 100000, n)
        })

        meta = instrument_meta or {}
        df["symbol"] = symbol
        df["market"] = meta.get("market", "synthetic")
        df["currency"] = meta.get("currency", "USD")
        df["unit"] = meta.get("unit", "unknown")
        df["provider"] = "synthetic"
        df["source"] = f"synthetic:{symbol}"
        df["retrieval_timestamp"] = datetime.now(timezone.utc).isoformat()
        df["timezone"] = "UTC"

        return df
