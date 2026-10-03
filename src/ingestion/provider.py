import logging
import time
import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MarketDataProvider(ABC):
    """Abstract base class for all market data providers."""

    @abstractmethod
    def get_spot_prices(self, symbols: List[str]) -> pd.DataFrame:
        """Fetch latest spot prices for given symbols."""
        pass

    @abstractmethod
    def get_historical_prices(
        self, symbol: str, start_date: str, end_date: str, interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch historical price data for a single symbol."""
        pass

    @abstractmethod
    def get_currency_prices(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch exchange rates (e.g., USD/INR)."""
        pass

    @abstractmethod
    def get_macro_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch macro metrics like treasury yields or indices."""
        pass


class YFinanceDataProvider(MarketDataProvider):
    """
    Implementation of MarketDataProvider using yfinance, with built-in retries,
    exponential backoff, rate limiting, and fallback synthetic data generator
    for offline execution and tests.
    """

    def __init__(self, max_retries: int = 3, backoff_factor: float = 1.0, timeout: int = 10):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.timeout = timeout

    def _fetch_yf_download(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Execute yfinance download with retries and exponential backoff."""
        for attempt in range(1, self.max_retries + 1):
            try:
                import yfinance as yf
                df = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=False,
                    timeout=self.timeout,
                )
                if df is not None and not df.empty:
                    # Flatten multi-index columns if returned by yfinance
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df.reset_index(inplace=True)
                    # Normalize column names
                    col_map = {col: str(col).capitalize() for col in df.columns}
                    df.rename(columns=col_map, inplace=True)
                    if "Date" in df.columns:
                        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
                    return df
            except Exception as e:
                logger.warning(f"Attempt {attempt} failed fetching {ticker}: {e}")
                if attempt < self.max_retries:
                    time.sleep(self.backoff_factor * (2 ** (attempt - 1)))

        logger.info(f"Using synthetic data fallback for {ticker} from {start_date} to {end_date}")
        return self._generate_synthetic_data(ticker, start_date, end_date)

    def _generate_synthetic_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate realistic synthetic price data for testing/offline support."""
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        if len(dates) == 0:
            dates = pd.date_range(end=end_date, periods=30, freq="B")

        np.random.seed(abs(hash(ticker)) % (2**31 - 1))

        base_prices = {
            "GC=F": 2000.0,
            "SI=F": 24.0,
            "INR=X": 83.0,
            "DX-Y.NYB": 103.0,
            "^TNX": 4.2,
            "^IRX": 5.0,
            "CL=F": 75.0,
            "^GSPC": 5000.0,
            "^NSEI": 22000.0,
            "^VIX": 15.0,
            "^INDIAVIX": 13.0,
            "GOLDBEES.NS": 60.0,
            "SILVERBEES.NS": 75.0,
        }

        base_p = base_prices.get(ticker, 100.0)
        daily_returns = np.random.normal(0.0003, 0.012, size=len(dates))
        price_series = base_p * np.exp(np.cumsum(daily_returns))

        data = []
        for d, close_p in zip(dates, price_series):
            high_p = close_p * (1.0 + abs(np.random.normal(0, 0.005)))
            low_p = close_p * (1.0 - abs(np.random.normal(0, 0.005)))
            open_p = low_p + (high_p - low_p) * np.random.uniform(0.2, 0.8)
            vol = int(np.random.uniform(10000, 100000))
            data.append({
                "Date": d,
                "Open": round(open_p, 4),
                "High": round(high_p, 4),
                "Low": round(low_p, 4),
                "Close": round(close_p, 4),
                "Adj Close": round(close_p, 4),
                "Volume": vol,
            })

        return pd.DataFrame(data)

    def get_spot_prices(self, symbols: List[str]) -> pd.DataFrame:
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=5)
        results = []
        for sym in symbols:
            df = self._fetch_yf_download(sym, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
            if not df.empty:
                last_row = df.iloc[-1].to_dict()
                last_row["Symbol"] = sym
                results.append(last_row)
        return pd.DataFrame(results)

    def get_historical_prices(
        self, symbol: str, start_date: str, end_date: str, interval: str = "1d"
    ) -> pd.DataFrame:
        return self._fetch_yf_download(symbol, start_date, end_date)

    def get_currency_prices(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self._fetch_yf_download(symbol, start_date, end_date)

    def get_macro_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self._fetch_yf_download(symbol, start_date, end_date)

    def get_economic_indicators(self) -> pd.DataFrame:
        """Return scheduled macro economic events/calendar mock data."""
        events = [
            {"date": "2026-03-01", "event": "FOMC Rate Decision", "type": "central_bank", "importance": "high"},
            {"date": "2026-03-05", "event": "US CPI Release", "type": "cpi", "importance": "high"},
            {"date": "2026-03-10", "event": "US Nonfarm Payrolls", "type": "employment", "importance": "high"},
            {"date": "2026-03-15", "event": "RBI Policy Decision", "type": "central_bank", "importance": "high"},
        ]
        return pd.DataFrame(events)
