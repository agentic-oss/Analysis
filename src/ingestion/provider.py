"""
Market data provider interface and implementations.
"""

from abc import ABC, abstractmethod
import time
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

logger = logging.getLogger(__name__)


class MarketDataProvider(ABC):
    """Abstract base class for market data providers."""

    @abstractmethod
    def get_historical_prices(
        self,
        ticker: str,
        start_date: str = None,
        end_date: str = None,
        period: str = "max",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data for a ticker."""
        pass


class YFinanceDataProvider(MarketDataProvider):
    """Yahoo Finance data provider with retry mechanism and error handling."""

    def __init__(self, max_retries: int = 3, timeout: int = 30, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.timeout = timeout
        self.backoff_factor = backoff_factor

    def get_historical_prices(
        self,
        ticker: str,
        start_date: str = None,
        end_date: str = None,
        period: str = "max",
        interval: str = "1d",
    ) -> pd.DataFrame:
        attempt = 0
        while attempt < self.max_retries:
            try:
                logger.info(f"Fetching data for {ticker} (attempt {attempt + 1})")
                t = yf.Ticker(ticker)
                if start_date and end_date:
                    df = t.history(start=start_date, end=end_date, interval=interval, timeout=self.timeout)
                elif start_date:
                    df = t.history(start=start_date, interval=interval, timeout=self.timeout)
                else:
                    df = t.history(period=period, interval=interval, timeout=self.timeout)

                if df.empty:
                    logger.warning(f"Empty dataframe returned for ticker {ticker}")
                    return pd.DataFrame()

                # Clean column names and index
                df = df.reset_index()
                # Standardize columns
                col_map = {
                    "Date": "date",
                    "Datetime": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Adj Close": "adj_close",
                    "Volume": "volume",
                }
                df = df.rename(columns=col_map)

                # Format date column to date string YYYY-MM-DD
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

                cols_to_keep = [c for c in ["date", "open", "high", "low", "close", "adj_close", "volume"] if c in df.columns]
                df = df[cols_to_keep]

                # Drop duplicates by date
                df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
                return df

            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {str(e)}")
                attempt += 1
                if attempt < self.max_retries:
                    sleep_time = self.backoff_factor ** attempt
                    time.sleep(sleep_time)
                else:
                    raise RuntimeError(f"Failed to fetch data for {ticker} after {self.max_retries} retries: {str(e)}")

        return pd.DataFrame()


class IndianPriceConverter:
    """Helper class for explicit Indian price representations and currency conversions."""

    # 1 troy ounce = 31.1034768 grams
    OUNCE_TO_GRAMS = 31.1034768

    @staticmethod
    def convert_usd_oz_to_inr_10g(usd_price: float, usdinr_rate: float) -> float:
        """Convert USD/oz price to INR/10g."""
        if usd_price is None or usdinr_rate is None or np.isnan(usd_price) or np.isnan(usdinr_rate):
            return np.nan
        price_per_gram_usd = usd_price / IndianPriceConverter.OUNCE_TO_GRAMS
        price_per_gram_inr = price_per_gram_usd * usdinr_rate
        return price_per_gram_inr * 10.0

    @staticmethod
    def convert_usd_oz_to_inr_kg(usd_price: float, usdinr_rate: float) -> float:
        """Convert USD/oz price to INR/kg."""
        if usd_price is None or usdinr_rate is None or np.isnan(usd_price) or np.isnan(usdinr_rate):
            return np.nan
        price_per_gram_usd = usd_price / IndianPriceConverter.OUNCE_TO_GRAMS
        price_per_gram_inr = price_per_gram_usd * usdinr_rate
        return price_per_gram_inr * 1000.0
