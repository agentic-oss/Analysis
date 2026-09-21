import abc
import time
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MarketDataProvider(abc.ABC):
    """Abstract base class for all market data providers."""

    @abc.abstractmethod
    def fetch_historical_ohlcv(
        self,
        symbol: str,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch historical daily OHLCV data for a given ticker."""
        pass

    @abc.abstractmethod
    def fetch_latest_quote(
        self,
        symbol: str,
        ticker: str
    ) -> Dict[str, Any]:
        """Fetch latest quote / daily observation for a symbol."""
        pass


class YahooMarketDataProvider(MarketDataProvider):
    """Yahoo Finance provider implementation with retry and rate-limit support."""

    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        timeout: int = 15,
        request_delay: float = 0.5
    ):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.timeout = timeout
        self.request_delay = request_delay

    def _execute_with_retry(self, func, *args, **kwargs):
        last_exception = None
        for attempt in range(1, self.max_retries + 1):
            try:
                time.sleep(self.request_delay)
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                wait_time = self.backoff_factor ** attempt
                logger.warning(
                    f"Attempt {attempt}/{self.max_retries} failed for {func.__name__}: {e}. Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
        raise RuntimeError(f"Failed after {self.max_retries} retries: {last_exception}")

    def fetch_historical_ohlcv(
        self,
        symbol: str,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        import yfinance as yf

        def _fetch():
            t = yf.Ticker(ticker)
            df = t.history(start=start_date, end=end_date, interval=interval, auto_adjust=False)
            if df.empty:
                logger.warning(f"No data returned for symbol={symbol}, ticker={ticker}")
                return pd.DataFrame()

            df = df.reset_index()
            # Standardize column names
            col_map = {
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Adj Close': 'adj_close',
                'Volume': 'volume'
            }
            df = df.rename(columns=col_map)
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            df['symbol'] = symbol
            df['ticker'] = ticker
            df['provider'] = 'yahoo'
            df['source'] = 'yfinance'
            df['retrieval_timestamp'] = datetime.now(timezone.utc).isoformat()

            # Ensure required numeric columns exist
            for col in ['open', 'high', 'low', 'close', 'adj_close', 'volume']:
                if col not in df.columns:
                    df[col] = np.nan

            return df[['date', 'symbol', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 'provider', 'source', 'retrieval_timestamp']]

        try:
            return self._execute_with_retry(_fetch)
        except Exception as e:
            logger.error(f"Error fetching yfinance data for {symbol} ({ticker}): {e}")
            return pd.DataFrame()

    def fetch_latest_quote(
        self,
        symbol: str,
        ticker: str
    ) -> Dict[str, Any]:
        df = self.fetch_historical_ohlcv(
            symbol, ticker,
            start_date=(pd.Timestamp.now() - pd.Timedelta(days=7)).strftime('%Y-%m-%d'),
            end_date=(pd.Timestamp.now() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        )
        if df.empty:
            return {}
        last_row = df.iloc[-1].to_dict()
        return last_row


class MockMarketDataProvider(MarketDataProvider):
    """Mock/Fallback provider generating deterministic synthetic daily data for testing/offline use."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fetch_historical_ohlcv(
        self,
        symbol: str,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        if len(dates) == 0:
            return pd.DataFrame()

        np.random.seed(self.seed + hash(symbol) % 1000)

        # Base prices depending on symbol
        base_price_map = {
            'GOLD': 2500.0,
            'SILVER': 30.0,
            'GOLD_FUTURES': 2510.0,
            'SILVER_FUTURES': 30.2,
            'GOLD_INR': 75000.0,
            'SILVER_INR': 88000.0,
            'MCX_GOLD': 75100.0,
            'MCX_SILVER': 88200.0,
            'USDINR': 83.5,
            'DXY': 103.0,
            'US10Y': 4.2,
            'US2Y': 4.5,
            'CRUDE_OIL': 75.0,
            'BRENT_CRUDE': 78.0,
            'SP500': 5500.0,
            'NIFTY50': 24000.0,
            'VIX': 15.0,
            'INDIA_VIX': 14.0
        }

        base_price = base_price_map.get(symbol, 100.0)
        returns = np.random.normal(0.0003, 0.012, len(dates))
        price_path = base_price * np.exp(np.cumsum(returns))

        records = []
        for i, dt in enumerate(dates):
            close = price_path[i]
            open_p = close * (1 + np.random.normal(0, 0.003))
            high = max(open_p, close) * (1 + abs(np.random.normal(0, 0.005)))
            low = min(open_p, close) * (1 - abs(np.random.normal(0, 0.005)))
            volume = float(np.random.randint(10000, 500000))

            records.append({
                'date': dt.strftime('%Y-%m-%d'),
                'symbol': symbol,
                'ticker': ticker,
                'open': round(float(open_p), 4),
                'high': round(float(high), 4),
                'low': round(float(low), 4),
                'close': round(float(close), 4),
                'adj_close': round(float(close), 4),
                'volume': volume,
                'provider': 'mock',
                'source': 'synthetic',
                'retrieval_timestamp': datetime.now(timezone.utc).isoformat()
            })

        return pd.DataFrame(records)

    def fetch_latest_quote(
        self,
        symbol: str,
        ticker: str
    ) -> Dict[str, Any]:
        today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
        df = self.fetch_historical_ohlcv(symbol, ticker, today_str, today_str)
        if df.empty:
            return {}
        return df.iloc[-1].to_dict()
