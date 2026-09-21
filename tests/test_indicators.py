import pytest
import pandas as pd
import numpy as np
from src.indicators.technical import TechnicalIndicators
from src.ingestion.provider import MockMarketDataProvider


def test_technical_indicators_calculation():
    provider = MockMarketDataProvider()
    raw = provider.fetch_historical_ohlcv("GOLD", "GC=F", "2025-01-01", "2026-01-15")
    df = TechnicalIndicators.calculate_all(raw)

    assert "sma_20" in df.columns
    assert "sma_50" in df.columns
    assert "sma_200" in df.columns
    assert "rsi_14" in df.columns
    assert "macd" in df.columns
    assert "macd_hist" in df.columns
    assert "atr_14" in df.columns
    assert "volatility_20d" in df.columns
    assert "dist_sma_200" in df.columns

    rsi_last = df["rsi_14"].iloc[-1]
    assert 0 <= rsi_last <= 100
