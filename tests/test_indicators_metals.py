import numpy as np
import pandas as pd
import pytest
from src.indicators.technical import TechnicalIndicators
from src.metals.analytics import MetalCrossMarketAnalytics

def test_technical_indicators():
    dates = pd.date_range("2025-01-01", periods=250, freq="D")
    np.random.seed(42)
    prices = 2000 + np.cumsum(np.random.randn(250) * 10)

    df = pd.DataFrame({
        "date": dates,
        "open": prices - 2,
        "high": prices + 5,
        "low": prices - 5,
        "close": prices,
        "volume": 10000 + np.random.randint(0, 1000, 250)
    })

    res = TechnicalIndicators.calculate_all(df)

    assert "sma_20" in res.columns
    assert "rsi_14" in res.columns
    assert "macd" in res.columns
    assert "atr_14" in res.columns
    assert "bollinger_upper" in res.columns
    assert "dist_sma_200" in res.columns
    assert not res["rsi_14"].isna().all()

def test_rolling_correlations():
    dates = pd.date_range("2025-01-01", periods=100, freq="D")
    np.random.seed(42)

    gold_df = pd.DataFrame({"date": dates, "close": 2000 + np.cumsum(np.random.randn(100) * 5)})
    dxy_df = pd.DataFrame({"date": dates, "close": 100 + np.cumsum(np.random.randn(100))})

    res = MetalCrossMarketAnalytics.calculate_rolling_correlations(
        metal_df=gold_df,
        macro_datasets={"DXY": dxy_df},
        windows=[20]
    )

    assert "corr_dxy_20d" in res.columns
