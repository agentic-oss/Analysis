"""Yahoo Finance provider and resilient sample fallback data generator."""

import time
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

from src.ingestion.base import MarketDataProvider

logger = logging.getLogger(__name__)


class YahooFinanceProvider(MarketDataProvider):
    """Data provider using yfinance with exponential backoff and fallback sample generator."""

    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0, timeout_seconds: int = 15):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.timeout_seconds = timeout_seconds

    def _fetch_yf_df(self, ticker_symbol: str, start_date: str, end_date: str, symbol_label: str) -> pd.DataFrame:
        attempt = 0
        while attempt < self.max_retries:
            try:
                logger.info(f"Fetching {ticker_symbol} from yfinance (attempt {attempt+1})...")
                df = yf.download(
                    ticker_symbol,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=False,
                    timeout=self.timeout_seconds
                )
                if df is not None and not df.empty:
                    # Flatten multi-index columns if present
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df = df.reset_index()
                    df.columns = [str(col).lower() for col in df.columns]
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

                    df['symbol'] = symbol_label
                    df['provider'] = 'yfinance'
                    df['retrieval_timestamp'] = datetime.utcnow().isoformat()
                    return df
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed for {ticker_symbol}: {e}")
                attempt += 1
                if attempt < self.max_retries:
                    time.sleep(self.backoff_factor ** attempt)

        logger.warning(f"Failed to fetch {ticker_symbol} from yfinance after {self.max_retries} attempts. Generating fallback synthetic data.")
        return self._generate_fallback_data(symbol_label, start_date, end_date)

    def _generate_fallback_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Generates realistic synthetic OHLCV data for testing/fallback."""
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        if len(dates) == 0:
            dates = pd.date_range(end=datetime.now(), periods=100, freq='B')

        np.random.seed(abs(hash(symbol)) % (2**32))

        base_prices = {
            "GOLD": 2000.0,
            "SILVER": 24.0,
            "GOLD_INR": 62000.0,
            "SILVER_INR": 75000.0,
            "USDINR": 83.0,
            "DXY": 104.0,
            "US10Y": 4.2,
            "US2Y": 4.5,
            "CRUDE_OIL": 75.0,
            "BRENT_CRUDE": 80.0,
            "SP500": 5000.0,
            "NIFTY50": 22000.0,
            "VIX": 15.0,
            "INDIAVIX": 14.0
        }

        base_price = base_prices.get(symbol, 100.0)
        returns = np.random.normal(0.0003, 0.012, size=len(dates))
        price_path = base_price * np.exp(np.cumsum(returns))

        records = []
        for d, p in zip(dates, price_path):
            high = p * (1 + abs(np.random.normal(0, 0.005)))
            low = p * (1 - abs(np.random.normal(0, 0.005)))
            open_p = p * (1 + np.random.normal(0, 0.002))
            close_p = p
            vol = int(np.random.uniform(5000, 50000))

            records.append({
                'date': d.strftime('%Y-%m-%d'),
                'open': round(float(open_p), 2),
                'high': round(float(high), 2),
                'low': round(float(low), 2),
                'close': round(float(close_p), 2),
                'adj close': round(float(close_p), 2),
                'volume': vol,
                'symbol': symbol,
                'provider': 'synthetic_fallback',
                'retrieval_timestamp': datetime.utcnow().isoformat()
            })

        return pd.DataFrame(records)

    def get_spot_prices(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        ticker_map = {
            "GOLD": "GC=F",
            "SILVER": "SI=F",
            "GOLD_INR": "GOLDBEES.NS",
            "SILVER_INR": "SILVERBEES.NS"
        }
        ticker = ticker_map.get(symbol, symbol)
        return self._fetch_yf_df(ticker, start_date, end_date, symbol)

    def get_historical_prices(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self.get_spot_prices(symbol, start_date, end_date)

    def get_futures_prices(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self.get_spot_prices(symbol, start_date, end_date)

    def get_currency_prices(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        ticker_map = {"USDINR": "USDINR=X"}
        ticker = ticker_map.get(symbol, symbol)
        return self._fetch_yf_df(ticker, start_date, end_date, symbol)

    def get_macro_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        ticker_map = {
            "DXY": "DX-Y.NYB",
            "US10Y": "^TNX",
            "US2Y": "2YY=F",
            "CRUDE_OIL": "CL=F",
            "BRENT_CRUDE": "BZ=F",
            "SP500": "^GSPC",
            "NIFTY50": "^NSEI",
            "VIX": "^VIX",
            "INDIAVIX": "^INDIAVIX"
        }
        ticker = ticker_map.get(symbol, symbol)
        return self._fetch_yf_df(ticker, start_date, end_date, symbol)

    def get_economic_indicators(self, indicator: str, start_date: str, end_date: str) -> pd.DataFrame:
        # Fallback or proxy indicator using macro symbols
        return self.get_macro_data(indicator, start_date, end_date)

    def get_event_data(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Returns structured economic calendar and policy decision events."""
        events = [
            {
                "date": start_date,
                "event_type": "FOMC_DECISION",
                "central_bank": "Federal Reserve",
                "importance": "HIGH",
                "description": "FOMC Rate Decision & Policy Statement"
            },
            {
                "date": start_date,
                "event_type": "RBI_POLICY",
                "central_bank": "Reserve Bank of India",
                "importance": "HIGH",
                "description": "RBI Monetary Policy Committee Meeting"
            },
            {
                "date": start_date,
                "event_type": "US_CPI",
                "central_bank": "US BLS",
                "importance": "HIGH",
                "description": "US Consumer Price Index Inflation Report"
            }
        ]
        return events
