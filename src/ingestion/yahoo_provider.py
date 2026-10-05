"""Yahoo Finance Market Data Provider implementation with retries, exponential backoff, and synthetic fallback."""

import logging
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

from src.ingestion.provider import BaseMarketDataProvider, MarketObservation

logger = logging.getLogger(__name__)


class YahooMarketDataProvider(BaseMarketDataProvider):
    """Data provider using yfinance with retry/backoff and error handling."""

    def __init__(self, max_retries: int = 3, backoff_factor: float = 1.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def get_historical_prices(
        self,
        symbol: str,
        ticker: str,
        start_date: str,
        end_date: str,
        market: str = "spot",
        currency: str = "USD",
        unit: str = "oz"
    ) -> pd.DataFrame:
        """Fetch historical prices from yfinance with retry logic."""
        import yfinance as yf

        df = pd.DataFrame()
        for attempt in range(self.max_retries):
            try:
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=False
                )
                if isinstance(data, pd.DataFrame) and not data.empty:
                    # Handle multi-level columns if present
                    if isinstance(data.columns, pd.MultiIndex):
                        data = data.xs(ticker, axis=1, level=1) if ticker in data.columns.levels[1] else data.droplevel(1, axis=1)

                    df = data.reset_index()
                    # Standardize column names
                    col_map = {
                        'Date': 'date', 'Open': 'open', 'High': 'high',
                        'Low': 'low', 'Close': 'close', 'Adj Close': 'adj_close',
                        'Volume': 'volume'
                    }
                    df = df.rename(columns=col_map)
                    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                    df['symbol'] = symbol
                    df['provider'] = 'yahoo'
                    df['source'] = ticker
                    df['retrieval_timestamp'] = datetime.now(timezone.utc).isoformat()
                    df['market'] = market
                    df['currency'] = currency
                    df['unit'] = unit
                    df['timezone'] = 'UTC'
                    cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'provider', 'source', 'retrieval_timestamp', 'market', 'currency', 'unit', 'timezone']
                    existing_cols = [c for c in cols if c in df.columns]
                    df = df[existing_cols]
                    return df
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {ticker}: {e}")
                time.sleep(self.backoff_factor * (2 ** attempt))

        logger.error(f"Failed to fetch data for {ticker} from Yahoo Finance after {self.max_retries} attempts.")
        return pd.DataFrame()

    def get_spot_prices(self, symbol: str, ticker: str) -> Optional[MarketObservation]:
        """Fetch latest single price observation from yfinance."""
        import yfinance as yf

        for attempt in range(self.max_retries):
            try:
                t = yf.Ticker(ticker)
                hist = t.history(period="5d")
                if not hist.empty:
                    latest = hist.iloc[-1]
                    date_str = hist.index[-1].strftime("%Y-%m-%d")
                    return MarketObservation(
                        date=date_str,
                        symbol=symbol,
                        open_price=float(latest.get("Open", 0.0)),
                        high_price=float(latest.get("High", 0.0)),
                        low_price=float(latest.get("Low", 0.0)),
                        close_price=float(latest.get("Close", 0.0)),
                        volume=float(latest.get("Volume", 0.0)),
                        provider="yahoo",
                        source=ticker,
                        retrieval_timestamp=datetime.now(timezone.utc).isoformat(),
                        market="spot",
                        currency="USD",
                        unit="oz"
                    )
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed fetching spot for {ticker}: {e}")
                time.sleep(self.backoff_factor * (2 ** attempt))
        return None


class SyntheticMarketDataProvider(BaseMarketDataProvider):
    """
    Deterministic/Synthetic data provider for testing, offline execution,
    and historical backfilling when live network data is unavailable.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def get_historical_prices(
        self,
        symbol: str,
        ticker: str,
        start_date: str,
        end_date: str,
        market: str = "spot",
        currency: str = "USD",
        unit: str = "oz"
    ) -> pd.DataFrame:
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        n = len(dates)
        if n == 0:
            return pd.DataFrame()

        np.random.seed(self.seed + hash(symbol) % 10000)

        # Set baseline base prices
        base_prices = {
            'GOLD': 2000.0, 'GOLD_FUT': 2010.0, 'SILVER': 24.0, 'SILVER_FUT': 24.2,
            'USDINR': 83.0, 'DXY': 104.0, 'US10Y': 4.2, 'US2Y': 4.5,
            'US_REAL_YIELD': 1.8, 'US_CPI': 305.0, 'IN_CPI': 180.0,
            'WTI': 75.0, 'BRENT': 80.0, 'SP500': 5000.0, 'NIFTY50': 22000.0,
            'VIX': 15.0, 'INDIA_VIX': 13.0, 'GOLD_INR': 65000.0, 'SILVER_INR': 75000.0,
            'MCX_GOLD': 65200.0, 'MCX_SILVER': 75500.0
        }

        start_p = base_prices.get(symbol, 100.0)
        daily_returns = np.random.normal(0.0002, 0.01, n)
        prices = start_p * np.exp(np.cumsum(daily_returns))

        records = []
        for i, dt in enumerate(dates):
            p = prices[i]
            high = p * (1 + abs(np.random.normal(0, 0.005)))
            low = p * (1 - abs(np.random.normal(0, 0.005)))
            op = p * (1 + np.random.normal(0, 0.002))
            vol = float(np.random.randint(1000, 50000))

            records.append({
                'date': dt.strftime('%Y-%m-%d'),
                'symbol': symbol,
                'open': round(float(op), 4),
                'high': round(float(high), 4),
                'low': round(float(low), 4),
                'close': round(float(p), 4),
                'volume': vol,
                'provider': 'synthetic',
                'source': ticker,
                'retrieval_timestamp': datetime.now(timezone.utc).isoformat(),
                'market': market,
                'currency': currency,
                'unit': unit,
                'timezone': 'UTC'
            })

        return pd.DataFrame(records)

    def get_spot_prices(self, symbol: str, ticker: str) -> Optional[MarketObservation]:
        df = self.get_historical_prices(symbol, ticker, "2026-01-01", "2026-01-05")
        if df.empty:
            return None
        row = df.iloc[-1]
        return MarketObservation(
            date=row['date'],
            symbol=symbol,
            open_price=row['open'],
            high_price=row['high'],
            low_price=row['low'],
            close_price=row['close'],
            volume=row['volume'],
            provider='synthetic',
            source=ticker,
            retrieval_timestamp=datetime.now(timezone.utc).isoformat(),
            market=row['market'],
            currency=row['currency'],
            unit=row['unit']
        )
