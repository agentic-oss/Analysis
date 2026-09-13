import datetime
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Any, Optional

from src.ingestion.provider import MarketDataProvider, retry_with_backoff

logger = logging.getLogger(__name__)

class YahooFinanceProvider(MarketDataProvider):
    """Data provider using yfinance."""

    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    @retry_with_backoff(max_retries=3, backoff_factor=1.5)
    def fetch_historical(
        self,
        symbol: str,
        ticker: str,
        start_date: str = "2015-01-01",
        end_date: Optional[str] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        if end_date is None:
            end_date = datetime.date.today().strftime("%Y-%m-%d")

        logger.info(f"Fetching {symbol} ({ticker}) from yfinance from {start_date} to {end_date}...")
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False, timeout=self.timeout)

        if df.empty:
            logger.warning(f"No data retrieved for {symbol} ({ticker})")
            return pd.DataFrame()

        # Handle MultiIndex columns if returned by recent yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        # Standardize date column
        date_col = 'Date' if 'Date' in df.columns else df.columns[0]
        df['Date'] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
        df = df.set_index('Date')

        # Ensure required columns
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col not in df.columns:
                df[col] = df['Close'] if 'Close' in df.columns else np.nan

        res_df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        res_df['provider'] = 'yfinance'
        res_df['symbol'] = symbol
        res_df['retrieval_timestamp'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return res_df

    def fetch_latest(self, symbol: str, ticker: str) -> Dict[str, Any]:
        df = self.fetch_historical(symbol, ticker, start_date=(datetime.date.today() - datetime.timedelta(days=7)).strftime("%Y-%m-%d"))
        if df.empty:
            return {}
        latest = df.iloc[-1]
        return {
            "symbol": symbol,
            "date": df.index[-1].strftime("%Y-%m-%d"),
            "open": float(latest['Open']),
            "high": float(latest['High']),
            "low": float(latest['Low']),
            "close": float(latest['Close']),
            "volume": float(latest['Volume']),
            "retrieval_timestamp": latest['retrieval_timestamp']
        }


class MockMarketDataProvider(MarketDataProvider):
    """Deterministic offline synthetic data provider for testing/offline use."""

    def __init__(self, base_date: str = "2015-01-01"):
        self.base_date = base_date

    def fetch_historical(
        self,
        symbol: str,
        ticker: str,
        start_date: str = "2015-01-01",
        end_date: Optional[str] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        if end_date is None:
            end_date = datetime.date.today().strftime("%Y-%m-%d")

        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        if len(dates) == 0:
            return pd.DataFrame()

        seed_val = abs(hash(symbol)) % (2**32)
        np.random.seed(seed_val)

        base_prices = {
            "GOLD": 1800.0,
            "SILVER": 22.0,
            "GOLD_FUTURES": 1805.0,
            "SILVER_FUTURES": 22.1,
            "USDINR": 82.5,
            "DXY": 103.0,
            "US10Y": 3.8,
            "US2Y": 4.2,
            "REAL_YIELD": 1.5,
            "CRUDE_OIL": 75.0,
            "BRENT_CRUDE": 78.0,
            "SP500": 4200.0,
            "NASDAQ": 13000.0,
            "DOW": 34000.0,
            "NIFTY50": 18000.0,
            "VIX": 18.0,
            "INDIAVIX": 15.0,
            "GOLD_INR": 58000.0,
            "SILVER_INR": 70000.0,
            "MCX_GOLD": 58500.0,
            "MCX_SILVER": 70500.0
        }

        start_price = base_prices.get(symbol, 100.0)
        returns = np.random.normal(0.0003, 0.012, size=len(dates))
        price_series = start_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame(index=dates)
        df['Close'] = price_series
        df['Open'] = df['Close'] * (1 + np.random.normal(0, 0.003, len(dates)))
        df['High'] = np.maximum(df['Open'], df['Close']) * (1 + np.abs(np.random.normal(0, 0.005, len(dates))))
        df['Low'] = np.minimum(df['Open'], df['Close']) * (1 - np.abs(np.random.normal(0, 0.005, len(dates))))
        df['Volume'] = np.random.randint(1000, 100000, size=len(dates))
        df['provider'] = 'mock'
        df['symbol'] = symbol
        df['retrieval_timestamp'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return df

    def fetch_latest(self, symbol: str, ticker: str) -> Dict[str, Any]:
        df = self.fetch_historical(symbol, ticker, start_date=(datetime.date.today() - datetime.timedelta(days=7)).strftime("%Y-%m-%d"))
        latest = df.iloc[-1]
        return {
            "symbol": symbol,
            "date": df.index[-1].strftime("%Y-%m-%d"),
            "open": float(latest['Open']),
            "high": float(latest['High']),
            "low": float(latest['Low']),
            "close": float(latest['Close']),
            "volume": float(latest['Volume']),
            "retrieval_timestamp": latest['retrieval_timestamp']
        }
