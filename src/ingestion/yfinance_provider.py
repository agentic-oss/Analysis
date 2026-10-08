"""
YFinance implementation of MarketDataProvider.
"""
import time
import datetime
from typing import Dict, List, Optional, Any
import logging
import pandas as pd
import yfinance as yf

from src.ingestion.provider import MarketDataProvider

logger = logging.getLogger(__name__)


class YFinanceProvider(MarketDataProvider):
    """
    Data provider using yfinance with retries, exponential backoff, rate limiting, and standardisation.
    """

    def __init__(self, max_retries: int = 3, backoff_factor: float = 1.5, timeout_seconds: int = 15):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.timeout_seconds = timeout_seconds

    def _fetch_ticker_history(
        self, ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch raw ticker history with retry logic.
        """
        delay = 1.0
        for attempt in range(1, self.max_retries + 1):
            try:
                t = yf.Ticker(ticker)
                if start_date and end_date:
                    # Adjust end_date +1 day for yfinance exclusive boundary
                    dt_end = pd.to_datetime(end_date) + pd.Timedelta(days=1)
                    df = t.history(start=start_date, end=dt_end.strftime("%Y-%m-%d"), auto_adjust=False)
                elif start_date:
                    df = t.history(start=start_date, auto_adjust=False)
                else:
                    df = t.history(period="max", auto_adjust=False)

                if isinstance(df, pd.DataFrame) and not df.empty:
                    return df

                logger.warning(f"Attempt {attempt}: Empty response for {ticker}")
            except Exception as e:
                logger.warning(f"Attempt {attempt} failed for {ticker}: {str(e)}")

            if attempt < self.max_retries:
                time.sleep(delay)
                delay *= self.backoff_factor

        return pd.DataFrame()

    def get_historical_prices(
        self,
        symbol_info: Dict[str, Any],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical daily prices and format according to schema.
        """
        symbol = symbol_info.get("symbol", "")
        ticker = symbol_info.get("ticker", symbol)
        currency = symbol_info.get("currency", "USD")
        unit = symbol_info.get("unit", "oz")

        df = self._fetch_ticker_history(ticker, start_date, end_date)
        if df.empty:
            return pd.DataFrame()

        df = df.reset_index()
        # Clean column names
        df.columns = [str(col).lower().replace(" ", "_") for col in df.columns]

        # Standardize date
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d")
        else:
            return pd.DataFrame()

        # Map required price columns
        col_map = {"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}
        for req_col in col_map.keys():
            if req_col not in df.columns:
                df[req_col] = df["close"] if "close" in df.columns else 0.0

        retrieval_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

        df["symbol"] = symbol
        df["currency"] = currency
        df["unit"] = unit
        df["provider"] = "yfinance"
        df["retrieval_timestamp"] = retrieval_ts

        target_cols = [
            "date",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "currency",
            "unit",
            "provider",
            "retrieval_timestamp",
        ]
        df = df[target_cols].sort_values("date").drop_duplicates(subset=["date"], keep="last")
        return df

    def get_spot_prices(self, instruments_config: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Get latest snapshot for instruments.
        """
        spot_dict = {}
        for inst in instruments_config:
            if not inst.get("enabled", True):
                continue
            df = self.get_historical_prices(inst)
            if not df.empty:
                last_row = df.iloc[-1]
                spot_dict[inst["symbol"]] = {
                    "symbol": inst["symbol"],
                    "date": last_row["date"],
                    "close": float(last_row["close"]),
                    "currency": last_row["currency"],
                    "unit": last_row["unit"],
                    "retrieval_timestamp": last_row["retrieval_timestamp"],
                }
        return spot_dict

    def fetch_all_instruments_data(
        self,
        instruments_config: List[Dict[str, Any]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all enabled instruments in config.
        """
        data = {}
        for inst in instruments_config:
            if not inst.get("enabled", True):
                continue
            sym = inst["symbol"]
            logger.info(f"Fetching data for {sym} ({inst.get('ticker')})...")
            df = self.get_historical_prices(inst, start_date, end_date)
            if not df.empty:
                data[sym] = df
        return data
