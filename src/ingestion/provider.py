"""Market Data Provider interface and data models."""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import pandas as pd


class MarketObservation:
    """Represents a single market data observation."""
    def __init__(
        self,
        date: str,
        symbol: str,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: float,
        provider: str,
        source: str,
        retrieval_timestamp: str,
        market: str,
        currency: str,
        unit: str,
        timezone_str: str = "UTC"
    ):
        self.date = date
        self.symbol = symbol
        self.open = open_price
        self.high = high_price
        self.low = low_price
        self.close = close_price
        self.volume = volume
        self.provider = provider
        self.source = source
        self.retrieval_timestamp = retrieval_timestamp
        self.market = market
        self.currency = currency
        self.unit = unit
        self.timezone = timezone_str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "symbol": self.symbol,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "provider": self.provider,
            "source": self.source,
            "retrieval_timestamp": self.retrieval_timestamp,
            "market": self.market,
            "currency": self.currency,
            "unit": self.unit,
            "timezone": self.timezone,
        }


class BaseMarketDataProvider(ABC):
    """Abstract base class for all market data providers."""

    @abstractmethod
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
        """Fetch historical daily OHLCV prices for a symbol/ticker."""
        pass

    @abstractmethod
    def get_spot_prices(self, symbol: str, ticker: str) -> Optional[MarketObservation]:
        """Fetch current spot/latest price observation."""
        pass
