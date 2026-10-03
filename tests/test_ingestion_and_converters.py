import pytest
import pandas as pd
from src.ingestion.provider import YFinanceDataProvider
from src.ingestion.indian_market import (
    convert_usd_oz_to_inr_10g,
    convert_usd_oz_to_inr_kg,
    process_indian_price_dataframe,
)

def test_unit_conversions():
    # Gold: USD 2000/oz at USDINR 83.0 -> 53,370.24 INR/10g
    gold_inr = convert_usd_oz_to_inr_10g(2000.0, 83.0)
    assert gold_inr == 53370.24

    # Silver: USD 24/oz at USDINR 83.0 -> 64,044.29 INR/kg
    silver_inr = convert_usd_oz_to_inr_kg(24.0, 83.0)
    assert silver_inr == 64044.29

def test_provider_synthetic_fallback():
    provider = YFinanceDataProvider()
    df = provider.get_historical_prices("GC=F", "2026-01-01", "2026-01-15")
    assert not df.empty
    assert "Close" in df.columns
    assert len(df) > 0

def test_indian_price_dataframe_processing():
    gold_df = pd.DataFrame({"Date": ["2026-01-01", "2026-01-02"], "Close": [2000.0, 2010.0]})
    silver_df = pd.DataFrame({"Date": ["2026-01-01", "2026-01-02"], "Close": [24.0, 24.5]})
    usdinr_df = pd.DataFrame({"Date": ["2026-01-01", "2026-01-02"], "Close": [83.0, 83.2]})

    g_proc, s_proc = process_indian_price_dataframe(gold_df, silver_df, usdinr_df)
    assert "Close_INR_10g" in g_proc.columns
    assert "Close_INR_kg" in s_proc.columns
    assert g_proc["Close_INR_10g"].iloc[0] == 53370.24
