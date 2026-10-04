import time
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class MarketDataProvider(ABC):
    """Abstract base class for precious metals and macro market data providers."""

    @abstractmethod
    def fetch_series(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        instrument_meta: Dict[str, Any]
    ) -> pd.DataFrame:
        """Fetch historical daily OHLCV series for a given ticker."""
        pass

    @abstractmethod
    def fetch_all_instruments(
        self,
        instruments: List[Dict[str, Any]],
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for all enabled instruments."""
        pass


class YFinanceDataProvider(MarketDataProvider):
    """yfinance market data provider with retries, exponential backoff, and metadata decoration."""

    def __init__(
        self,
        retries: int = 3,
        timeout: int = 10,
        backoff_factor: float = 1.5
    ):
        self.retries = retries
        self.timeout = timeout
        self.backoff_factor = backoff_factor

    def fetch_series(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        instrument_meta: Dict[str, Any]
    ) -> pd.DataFrame:
        import yfinance as yf

        retrieval_ts = datetime.now(timezone.utc).isoformat()
        last_exception = None

        for attempt in range(1, self.retries + 1):
            try:
                logger.info(f"Fetching {ticker} (attempt {attempt}/{self.retries})...")
                df = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=False,
                    timeout=self.timeout
                )

                if df.empty:
                    raise ValueError(f"No data returned for ticker {ticker}")

                # Flatten multi-index columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df = df.reset_index()
                # Ensure standard column names
                rename_map = {
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Adj Close": "adj_close",
                    "Volume": "volume"
                }
                df = df.rename(columns=rename_map)

                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

                # Decorate with metadata
                df["symbol"] = instrument_meta.get("symbol", ticker)
                df["ticker"] = ticker
                df["provider"] = "yfinance"
                df["source"] = "yahoo_finance"
                df["retrieval_timestamp"] = retrieval_ts
                df["market"] = instrument_meta.get("market", "")
                df["currency"] = instrument_meta.get("currency", "")
                df["unit"] = instrument_meta.get("unit", "")
                df["timezone"] = "UTC"

                # Standardize column selection
                cols = [
                    "date", "symbol", "ticker", "open", "high", "low", "close", "adj_close", "volume",
                    "provider", "source", "retrieval_timestamp", "market", "currency", "unit", "timezone"
                ]
                existing_cols = [c for c in cols if c in df.columns]
                return df[existing_cols]

            except Exception as e:
                last_exception = e
                logger.warning(f"Error fetching {ticker} on attempt {attempt}: {e}")
                if attempt < self.retries:
                    time.sleep(self.backoff_factor ** attempt)

        logger.error(f"Failed to fetch {ticker} after {self.retries} attempts.")
        raise RuntimeError(f"Data retrieval failed for {ticker}: {last_exception}")

    def fetch_all_instruments(
        self,
        instruments: List[Dict[str, Any]],
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        results = {}
        for inst in instruments:
            if not inst.get("enabled", True):
                continue
            symbol = inst["symbol"]
            ticker = inst["ticker"]
            try:
                results[symbol] = self.fetch_series(ticker, start_date, end_date, inst)
            except Exception as e:
                logger.error(f"Skipping {symbol} due to fetch error: {e}")
        return results


class MockDataProvider(MarketDataProvider):
    """Deterministic mock provider for offline execution, testing, and fallback."""

    def fetch_series(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        instrument_meta: Dict[str, Any]
    ) -> pd.DataFrame:
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        n = len(dates)
        if n == 0:
            return pd.DataFrame()

        np.random.seed(abs(hash(ticker)) % (2**32 - 1))

        # Set base prices based on instrument
        symbol = instrument_meta.get("symbol", ticker)
        if "GOLD" in symbol:
            base = 2000.0 if instrument_meta.get("currency") == "USD" else 62000.0
        elif "SILVER" in symbol:
            base = 25.0 if instrument_meta.get("currency") == "USD" else 72000.0
        elif symbol == "USDINR":
            base = 83.0
        elif symbol == "DXY":
            base = 104.0
        elif "10Y" in symbol or "2Y" in symbol:
            base = 4.2
        else:
            base = 100.0

        returns = np.random.normal(0.0002, 0.01, n)
        price_path = base * np.cumprod(1 + returns)

        df = pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "symbol": symbol,
            "ticker": ticker,
            "open": price_path * (1 + np.random.normal(0, 0.002, n)),
            "high": price_path * (1 + abs(np.random.normal(0, 0.005, n))),
            "low": price_path * (1 - abs(np.random.normal(0, 0.005, n))),
            "close": price_path,
            "adj_close": price_path,
            "volume": np.random.randint(1000, 500000, n),
            "provider": "mock",
            "source": "synthetic_generator",
            "retrieval_timestamp": datetime.now(timezone.utc).isoformat(),
            "market": instrument_meta.get("market", ""),
            "currency": instrument_meta.get("currency", ""),
            "unit": instrument_meta.get("unit", ""),
            "timezone": "UTC"
        })
        return df

    def fetch_all_instruments(
        self,
        instruments: List[Dict[str, Any]],
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        results = {}
        for inst in instruments:
            if not inst.get("enabled", True):
                continue
            symbol = inst["symbol"]
            ticker = inst["ticker"]
            results[symbol] = self.fetch_series(ticker, start_date, end_date, inst)
        return results
