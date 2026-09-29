import pandas as pd
import numpy as np
from typing import Dict, Any


def detect_macro_regime(macro_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Evaluates transparent rule-based composite macro regime:
    Returns regime label, composite score, and component signals.
    """
    components = {}

    # 1. Real Yields / Interest Rates
    us10y = macro_data.get("US10Y")
    real_yield = macro_data.get("US_REAL_YIELD")
    if real_yield is not None and not real_yield.empty:
        ry_change = real_yield["close"].iloc[-1] - real_yield["close"].iloc[-20] if len(real_yield) >= 20 else 0.0
        components["real_yield_change_20d"] = float(ry_change)
    else:
        components["real_yield_change_20d"] = 0.0

    # 2. DXY
    dxy = macro_data.get("DXY")
    if dxy is not None and not dxy.empty:
        dxy_ret = dxy["close"].iloc[-1] / dxy["close"].iloc[-20] - 1.0 if len(dxy) >= 20 else 0.0
        components["dxy_return_20d"] = float(dxy_ret)
    else:
        components["dxy_return_20d"] = 0.0

    # 3. Volatility / Equity
    vix = macro_data.get("VIX")
    if vix is not None and not vix.empty:
        components["vix_level"] = float(vix["close"].iloc[-1])
    else:
        components["vix_level"] = 15.0

    # Rules
    ry_chg = components["real_yield_change_20d"]
    dxy_ret = components["dxy_return_20d"]
    vix_lvl = components["vix_level"]

    if vix_lvl > 22.0:
        regime = "Risk-Off"
    elif ry_chg > 0.25 and dxy_ret > 0.015:
        regime = "Tightening"
    elif ry_chg < -0.25 and dxy_ret < -0.015:
        regime = "Easing"
    elif ry_chg < 0 and dxy_ret < 0:
        regime = "Inflationary"
    elif ry_chg > 0 and dxy_ret < 0:
        regime = "Disinflationary"
    else:
        regime = "Neutral"

    return {
        "macro_regime": regime,
        "components": components
    }


def detect_market_regime(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detects asset market regime (Trend + Volatility).
    """
    if df is None or df.empty or "close" not in df.columns:
        return {"trend_regime": "Neutral", "volatility_regime": "Normal"}

    close = df["close"]
    sma_50 = df["sma_50"].iloc[-1] if "sma_50" in df.columns and not np.isnan(df["sma_50"].iloc[-1]) else close.iloc[-1]
    sma_200 = df["sma_200"].iloc[-1] if "sma_200" in df.columns and not np.isnan(df["sma_200"].iloc[-1]) else close.iloc[-1]

    dist_200 = (close.iloc[-1] - sma_200) / sma_200 if sma_200 != 0 else 0.0

    if dist_200 > 0.05 and close.iloc[-1] > sma_50:
        trend = "Strong Bullish Trend"
    elif dist_200 > 0.0:
        trend = "Bullish Trend"
    elif dist_200 < -0.05 and close.iloc[-1] < sma_50:
        trend = "Strong Bearish Trend"
    elif dist_200 < 0.0:
        trend = "Bearish Trend"
    else:
        trend = "Range"

    vol_20d = df["volatility_20d"].iloc[-1] if "volatility_20d" in df.columns and not np.isnan(df["volatility_20d"].iloc[-1]) else 15.0
    if vol_20d > 22.0:
        vol = "High Volatility"
    elif vol_20d < 10.0:
        vol = "Low Volatility"
    else:
        vol = "Normal Volatility"

    return {
        "trend_regime": trend,
        "volatility_regime": vol,
        "combined_regime": f"{trend} | {vol}"
    }
