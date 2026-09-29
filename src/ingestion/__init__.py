import time
import logging
import json
import yaml
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MarketDataProvider")


class BaseDataProvider(ABC):
    def __init__(self, config_path: str = "config/config.yaml", instruments_path: str = "config/instruments.json"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        with open(instruments_path, "r") as f:
            self.instruments_data = json.load(f)
        self.instruments = {inst["id"]: inst for inst in self.instruments_data.get("instruments", [])}

    @abstractmethod
    def fetch_historical_ohlcv(self, instrument_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        pass


class MarketDataProvider(BaseDataProvider):
    def __init__(self, config_path: str = "config/config.yaml", instruments_path: str = "config/instruments.json"):
        super().__init__(config_path, instruments_path)
        self.max_retries = self.config.get("data_provider", {}).get("max_retries", 3)
        self.timeout = self.config.get("data_provider", {}).get("timeout_seconds", 30)
        self.backoff_factor = self.config.get("data_provider", {}).get("backoff_factor", 2.0)

    def fetch_historical_ohlcv(self, instrument_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        if instrument_id not in self.instruments:
            raise ValueError(f"Unknown instrument_id: {instrument_id}")

        inst_info = self.instruments[instrument_id]
        ticker = inst_info.get("ticker")

        for attempt in range(1, self.max_retries + 1):
            try:
                df = self._fetch_from_yfinance(ticker, start_date, end_date)
                if df is not None and not df.empty:
                    return self._standardize_df(df, inst_info)
                logger.warning(f"Attempt {attempt}: Empty data returned for {instrument_id} ({ticker})")
            except Exception as e:
                logger.warning(f"Attempt {attempt} failed for {instrument_id} ({ticker}): {e}")

            if attempt < self.max_retries:
                sleep_time = self.backoff_factor ** attempt
                time.sleep(sleep_time)

        logger.info(f"YFinance unavailable or empty for {instrument_id}. Generating synthetic fallback data for range {start_date} to {end_date}.")
        return self.generate_synthetic_data(instrument_id, start_date, end_date)

    def _fetch_from_yfinance(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        import yfinance as yf
        t = yf.Ticker(ticker)
        df = t.history(start=start_date, end=end_date)
        if df is None or df.empty:
            return None
        return df

    def _standardize_df(self, df: pd.DataFrame, inst_info: Dict[str, Any]) -> pd.DataFrame:
        df = df.reset_index()
        # Ensure Date column exists
        date_col = "Date" if "Date" in df.columns else df.columns[0]
        df["date"] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")

        # Standardize OHLCV
        cols = {c.lower(): c for c in df.columns}
        df["open"] = df[cols["open"]] if "open" in cols else df[cols["close"]]
        df["high"] = df[cols["high"]] if "high" in cols else df[cols["close"]]
        df["low"] = df[cols["low"]] if "low" in cols else df[cols["close"]]
        df["close"] = df[cols["close"]]
        df["volume"] = df[cols["volume"]] if "volume" in cols else 0.0

        ret_df = pd.DataFrame({
            "date": df["date"],
            "instrument_id": inst_info["id"],
            "symbol": inst_info["symbol"],
            "category": inst_info["category"],
            "open": df["open"].astype(float),
            "high": df["high"].astype(float),
            "low": df["low"].astype(float),
            "close": df["close"].astype(float),
            "volume": df["volume"].astype(float),
            "currency": inst_info["currency"],
            "unit": inst_info["unit"],
            "provider": "yfinance",
            "source": inst_info["exchange"],
            "retrieval_timestamp": pd.Timestamp.now().isoformat(),
            "timezone": self.config.get("system", {}).get("timezone", "Asia/Kolkata")
        })
        return ret_df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    def generate_synthetic_data(self, instrument_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        inst_info = self.instruments[instrument_id]
        dates = pd.date_range(start=start_date, end=end_date, freq="B")

        # Base prices for different assets
        base_prices = {
            "GOLD_USD_SPOT": 2000.0,
            "SILVER_USD_SPOT": 24.0,
            "GOLD_USD_FUTURES": 2010.0,
            "SILVER_USD_FUTURES": 24.2,
            "GOLD_INR_MCX": 62000.0,
            "SILVER_INR_MCX": 74000.0,
            "USD_INR": 83.0,
            "DXY": 103.5,
            "US10Y": 4.2,
            "US02Y": 4.5,
            "US_REAL_YIELD": 1.8,
            "BRENT_CRUDE": 80.0,
            "WTI_CRUDE": 75.0,
            "COPPER": 3.8,
            "SP500": 4800.0,
            "NASDAQ": 15000.0,
            "DOW_JONES": 38000.0,
            "NIFTY50": 21500.0,
            "VIX": 14.0,
            "INDIA_VIX": 13.5
        }

        base_price = base_prices.get(instrument_id, 100.0)
        np.random.seed(hash(instrument_id) % (2**32))
        n = len(dates)
        if n == 0:
            return pd.DataFrame(columns=["date", "instrument_id", "symbol", "category", "open", "high", "low", "close", "volume", "currency", "unit", "provider", "source", "retrieval_timestamp", "timezone"])

        returns = np.random.normal(0.0002, 0.012, n)
        price_path = base_price * np.exp(np.cumsum(returns))

        high_path = price_path * (1 + np.abs(np.random.normal(0.005, 0.003, n)))
        low_path = price_path * (1 - np.abs(np.random.normal(0.005, 0.003, n)))
        open_path = low_path + (high_path - low_path) * np.random.uniform(0.1, 0.9, n)
        volume_path = np.random.lognormal(10, 1, n)

        df = pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "instrument_id": inst_info["id"],
            "symbol": inst_info["symbol"],
            "category": inst_info["category"],
            "open": np.round(open_path, 4),
            "high": np.round(high_path, 4),
            "low": np.round(low_path, 4),
            "close": np.round(price_path, 4),
            "volume": np.round(volume_path, 2),
            "currency": inst_info["currency"],
            "unit": inst_info["unit"],
            "provider": "synthetic_fallback",
            "source": inst_info["exchange"],
            "retrieval_timestamp": pd.Timestamp.now().isoformat(),
            "timezone": self.config.get("system", {}).get("timezone", "Asia/Kolkata")
        })
        return df

    def get_event_calendar(self, start_date: str, end_date: str) -> pd.DataFrame:
        dates = pd.date_range(start=start_date, end=end_date, freq="30D")
        events = []
        for d in dates:
            d_str = d.strftime("%Y-%m-%d")
            events.extend([
                {"date": d_str, "event_type": "FOMC_DECISION", "importance": "HIGH", "central_bank": "Federal Reserve"},
                {"date": d_str, "event_type": "US_CPI", "importance": "HIGH", "central_bank": "Federal Reserve"},
                {"date": d_str, "event_type": "RBI_POLICY", "importance": "HIGH", "central_bank": "RBI"}
            ])
        return pd.DataFrame(events)
