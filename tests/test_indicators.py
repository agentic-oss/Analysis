import pytest
import pandas as pd
from src.ingestion.synthetic_provider import SyntheticProvider
from src.indicators.technical import TechnicalAnalysis

def test_technical_indicators_calculation():
    provider = SyntheticProvider(seed=42)
    df = provider.get_spot_prices("GOLD", "2023-01-01", "2023-06-01")
    df_tech = TechnicalAnalysis.calculate_all(df)

    assert "sma_20" in df_tech.columns
    assert "sma_50" in df_tech.columns
    assert "rsi_14" in df_tech.columns
    assert "macd" in df_tech.columns
    assert "atr_14" in df_tech.columns
    assert "bollinger_upper" not in df_tech.columns or "bb_upper" in df_tech.columns
    assert "bb_upper" in df_tech.columns

    # Verify RSI values are within bounds
    rsi_vals = df_tech["rsi_14"].dropna()
    assert (rsi_vals >= 0.0).all() and (rsi_vals <= 100.0).all()
