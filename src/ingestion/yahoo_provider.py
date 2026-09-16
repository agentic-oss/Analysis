import logging
from datetime import datetime
from typing import Any, Dict, Optional
import pandas as pd
import yfinance as yf

from src.ingestion.provider import MarketDataProvider

logger = logging.getLogger(__name__)


class YahooMarketDataProvider(MarketDataProvider):
    """Yahoo Finance implementation of MarketDataProvider."""

    def __init__(self, max_retries: int = 3, retry_backoff: float = 2.0):
        super().__init__(
            name="YahooFinance", max_retries=max_retries, retry_backoff=retry_backoff
        )

    def _fetch_yf_history(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d",
        period: str = "max",
    ) -> pd.DataFrame:
        def _fetch():
            t = yf.Ticker(ticker)
            if start_date:
                df = t.history(start=start_date, end=end_date, interval=interval)
            else:
                df = t.history(period=period, interval=interval)
            return df

        return self.execute_with_retry(_fetch)

    def get_historical_prices(
        self,
        symbol: str,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        df = self._fetch_yf_history(
            ticker, start_date=start_date, end_date=end_date, interval=interval
        )
        if df.empty:
            logger.warning(f"No history found for ticker {ticker}")
            return pd.DataFrame()

        df = df.reset_index()
        # Standardize column names
        cols_map = {
            "Date": "date",
            "Datetime": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
        df = df.rename(columns={c: cols_map[c] for c in df.columns if c in cols_map})
        if "adj_close" not in df.columns and "close" in df.columns:
            df["adj_close"] = df["close"]

        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df["symbol"] = symbol
        df["ticker"] = ticker
        df["provider"] = self.name
        df["retrieval_timestamp"] = datetime.utcnow().isoformat()
        return df

    def get_spot_prices(self, symbol: str, ticker: str) -> Dict[str, Any]:
        df = self.get_historical_prices(symbol, ticker, interval="1d")
        if df.empty:
            return {}
        latest = df.iloc[-1].to_dict()
        return latest

    def get_futures_prices(
        self, symbol: str, ticker: str, start_date: Optional[str] = None
    ) -> pd.DataFrame:
        return self.get_historical_prices(symbol, ticker, start_date=start_date)

    def get_currency_prices(
        self, symbol: str, ticker: str, start_date: Optional[str] = None
    ) -> pd.DataFrame:
        return self.get_historical_prices(symbol, ticker, start_date=start_date)

    def get_macro_data(
        self, symbol: str, ticker: str, start_date: Optional[str] = None
    ) -> pd.DataFrame:
        return self.get_historical_prices(symbol, ticker, start_date=start_date)

    def get_economic_indicators(self) -> pd.DataFrame:
        # Placeholder/synthetic calendar events generator if live macro calendar API isn't supplied
        dates = pd.date_range(end=datetime.utcnow(), periods=10, freq="30D")
        events = []
        for d in dates:
            events.append(
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "event_type": "FOMC",
                    "importance": "high",
                    "currency": "USD",
                    "actual": None,
                    "consensus": None,
                }
            )
            events.append(
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "event_type": "CPI",
                    "importance": "high",
                    "currency": "USD",
                    "actual": None,
                    "consensus": None,
                }
            )
        return pd.DataFrame(events)
