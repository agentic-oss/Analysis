from abc import ABC, abstractmethod
import logging
import time
from datetime import datetime, timezone
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class MarketDataProvider(ABC):
    """Abstract base class for all market data providers."""

    @abstractmethod
    def get_spot_prices(self, symbols: list[str], start_date: str = None, end_date: str = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_historical_prices(self, symbols: list[str], start_date: str = None, end_date: str = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_futures_prices(self, symbols: list[str], start_date: str = None, end_date: str = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_currency_prices(self, symbols: list[str], start_date: str = None, end_date: str = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_macro_data(self, symbols: list[str], start_date: str = None, end_date: str = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_economic_indicators(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        pass


class YFinanceDataProvider(MarketDataProvider):
    """Implementation of MarketDataProvider using yfinance with retries and fallbacks."""

    def __init__(self, timeout_seconds: int = 15, max_retries: int = 3, retry_backoff_factor: float = 2.0):
        self.provider_name = "yfinance"
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_factor = retry_backoff_factor

    def _fetch_ticker_history(self, ticker: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Fetch historical data with retries, exponential backoff, and error logging."""
        import yfinance as yf

        retries = 0
        while retries < self.max_retries:
            try:
                t = yf.Ticker(ticker)
                if start_date and end_date:
                    df = t.history(start=start_date, end=end_date, auto_adjust=False)
                elif start_date:
                    df = t.history(start=start_date, auto_adjust=False)
                else:
                    df = t.history(period="max", auto_adjust=False)

                if df is not None and not df.empty:
                    df = df.reset_index()
                    if "Date" in df.columns:
                        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
                    return df
            except Exception as e:
                logger.warning(f"Attempt {retries + 1} failed for ticker {ticker}: {e}")
            retries += 1
            time.sleep(self.retry_backoff_factor ** retries)

        logger.error(f"Failed to fetch data for {ticker} after {self.max_retries} retries.")
        return pd.DataFrame()

    def fetch_symbol_data(self, instrument_info: dict, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Fetch and standardize data for a single instrument dict."""
        ticker = instrument_info.get("ticker", instrument_info.get("symbol"))
        symbol = instrument_info.get("symbol")
        df = self._fetch_ticker_history(ticker, start_date, end_date)

        if df.empty:
            return df

        # Standardize columns
        col_map = {
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume"
        }
        df = df.rename(columns=col_map)

        # Standardize metadata
        now_iso = datetime.now(timezone.utc).isoformat()
        df["symbol"] = symbol
        df["provider"] = self.provider_name
        df["source"] = f"yfinance:{ticker}"
        df["retrieval_timestamp"] = now_iso
        df["market"] = instrument_info.get("market", "unknown")
        df["currency"] = instrument_info.get("currency", "USD")
        df["unit"] = instrument_info.get("unit", "price")
        df["timezone"] = "UTC"

        for col in ["open", "high", "low", "close", "adj_close", "volume"]:
            if col not in df.columns:
                df[col] = np.nan

        return df[["date", "symbol", "open", "high", "low", "close", "adj_close", "volume",
                   "provider", "source", "retrieval_timestamp", "market", "currency", "unit", "timezone"]]

    def get_spot_prices(self, symbols: list[str], start_date: str = None, end_date: str = None) -> pd.DataFrame:
        dfs = []
        for sym in symbols:
            info = {"symbol": sym, "ticker": sym, "market": "spot"}
            df = self.fetch_symbol_data(info, start_date, end_date)
            if not df.empty:
                dfs.append(df)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def get_historical_prices(self, symbols: list[str], start_date: str = None, end_date: str = None) -> pd.DataFrame:
        return self.get_spot_prices(symbols, start_date, end_date)

    def get_futures_prices(self, symbols: list[str], start_date: str = None, end_date: str = None) -> pd.DataFrame:
        dfs = []
        for sym in symbols:
            info = {"symbol": sym, "ticker": sym, "market": "futures"}
            df = self.fetch_symbol_data(info, start_date, end_date)
            if not df.empty:
                dfs.append(df)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def get_currency_prices(self, symbols: list[str], start_date: str = None, end_date: str = None) -> pd.DataFrame:
        dfs = []
        for sym in symbols:
            info = {"symbol": sym, "ticker": sym, "market": "currency"}
            df = self.fetch_symbol_data(info, start_date, end_date)
            if not df.empty:
                dfs.append(df)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def get_macro_data(self, symbols: list[str], start_date: str = None, end_date: str = None) -> pd.DataFrame:
        dfs = []
        for sym in symbols:
            info = {"symbol": sym, "ticker": sym, "market": "macro"}
            df = self.fetch_symbol_data(info, start_date, end_date)
            if not df.empty:
                dfs.append(df)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def get_economic_indicators(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        # Returns event/indicator calendar framework
        return pd.DataFrame(columns=["date", "event_type", "event_importance", "consensus", "actual", "surprise"])
