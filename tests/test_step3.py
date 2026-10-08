import pytest
import pandas as pd
import numpy as np
from src.indicators.technical import TechnicalIndicators
from src.metals.gold import GoldAnalyzer, SilverAnalyzer
from src.ratios.gold_silver_ratio import GoldSilverRatio

def test_technical_indicators():
    dates = pd.date_range("2026-01-01", periods=30, freq="B").strftime("%Y-%m-%d")
    close_prices = np.linspace(100, 130, 30)
    df = pd.DataFrame({
        "date": dates,
        "open": close_prices - 1,
        "high": close_prices + 2,
        "low": close_prices - 2,
        "close": close_prices,
        "volume": [1000] * 30
    })

    df_ind = TechnicalIndicators.add_all_indicators(df)
    assert "sma_20" in df_ind.columns
    assert "rsi_14" in df_ind.columns
    assert "macd" in df_ind.columns
    assert "atr_14" in df_ind.columns
    assert "bollinger_upper" in df_ind.columns
    assert len(df_ind) == 30

def test_gold_silver_analyzers():
    gold_df = pd.DataFrame({"date": ["2026-03-20"], "close": [2500.0], "rsi_14": [60.0]})
    silver_df = pd.DataFrame({"date": ["2026-03-20"], "close": [30.0], "rsi_14": [55.0]})

    gold_res = GoldAnalyzer.analyze_gold(gold_df, usd_inr_rate=83.0)
    silver_res = SilverAnalyzer.analyze_silver(silver_df, usd_inr_rate=83.0)

    assert gold_res["symbol"] == "GOLD"
    assert gold_res["price_usd_oz"] == 2500.0
    assert gold_res["price_inr_10g"] > 0

    assert silver_res["symbol"] == "SILVER"
    assert silver_res["price_usd_oz"] == 30.0
    assert silver_res["price_inr_kg"] > 0

def test_gold_silver_ratio():
    gold_df = pd.DataFrame({"date": ["2026-03-20"], "close": [2400.0]})
    silver_df = pd.DataFrame({"date": ["2026-03-20"], "close": [30.0]})

    ratio_df = GoldSilverRatio.calculate_ratio(gold_df, silver_df)
    assert "gold_silver_ratio" in ratio_df.columns
    assert ratio_df["gold_silver_ratio"].iloc[0] == 80.0

    analysis = GoldSilverRatio.analyze_latest_ratio(ratio_df)
    assert analysis["current_ratio"] == 80.0
