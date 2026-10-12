import pytest
import pandas as pd
import numpy as np
from src.ingestion.provider import SyntheticMarketDataProvider
from src.indicators.technical import compute_all_technical_indicators
from src.metals.correlation import MultiAssetCorrelationAnalyzer
from src.ratios.gold_silver import GoldSilverRatioAnalyzer

def test_technical_indicators():
    provider = SyntheticMarketDataProvider(seed=42)
    df = provider.fetch_historical_series("GC=F", "2025-01-01", "2026-01-01")
    enriched = compute_all_technical_indicators(df)

    assert "rsi_14" in enriched.columns
    assert "macd" in enriched.columns
    assert "bb_upper" in enriched.columns
    assert "dist_sma_200" in enriched.columns
    assert len(enriched) == len(df)

def test_correlation_analyzer():
    provider = SyntheticMarketDataProvider(seed=42)
    gold_df = provider.fetch_historical_series("GC=F", "2025-01-01", "2026-01-01")
    dxy_df = provider.fetch_historical_series("DX-Y.NYB", "2025-01-01", "2026-01-01")

    g_ret = gold_df.set_index("date")["close"].pct_change()
    d_ret = dxy_df.set_index("date")["close"].pct_change()

    analyzer = MultiAssetCorrelationAnalyzer()
    corrs = analyzer.compute_correlations(g_ret, {"DXY": d_ret})
    assert "DXY" in corrs
    assert 20 in corrs["DXY"]

def test_gold_silver_ratio():
    provider = SyntheticMarketDataProvider(seed=42)
    gold_df = provider.fetch_historical_series("GC=F", "2025-01-01", "2026-01-01")
    silver_df = provider.fetch_historical_series("SI=F", "2025-01-01", "2026-01-01")

    analyzer = GoldSilverRatioAnalyzer()
    res = analyzer.analyze_ratio(gold_df, silver_df)
    assert "current_ratio" in res
    assert "z_score" in res
    assert res["current_ratio"] > 0
