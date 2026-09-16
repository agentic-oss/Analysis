from typing import Dict, Any, Optional
import pandas as pd


class RegimeDetector:
    """Classifies composite Macro Regimes and individual Market Regimes for Gold and Silver."""

    @staticmethod
    def detect_market_regime(df: pd.DataFrame) -> str:
        """Classifies technical market regime for a metal (e.g. Gold or Silver)."""
        if df.empty or "close" not in df.columns or "sma_50" not in df.columns:
            return "Neutral"

        latest = df.iloc[-1]
        close = latest["close"]
        sma_20 = latest.get("sma_20", close)
        sma_50 = latest.get("sma_50", close)
        sma_200 = latest.get("sma_200", close)
        vol_20 = latest.get("volatility_20d", 0.15)

        is_bullish = close > sma_50 and sma_20 > sma_50
        is_bearish = close < sma_50 and sma_20 < sma_50
        is_above_200 = close > sma_200

        if is_bullish and is_above_200:
            trend = "Strong Bullish"
        elif is_bullish:
            trend = "Bullish"
        elif is_bearish and not is_above_200:
            trend = "Strong Bearish"
        elif is_bearish:
            trend = "Bearish"
        else:
            trend = "Range"

        vol_regime = "High Volatility" if vol_20 > 0.25 else ("Low Volatility" if vol_20 < 0.12 else "Normal Volatility")
        return f"{trend} | {vol_regime}"

    @staticmethod
    def detect_macro_regime(macro_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Transparent rule-based composite macro regime classification."""
        dxy_df = macro_data.get("DXY", pd.DataFrame())
        rates_df = macro_data.get("US10Y", pd.DataFrame())
        vix_df = macro_data.get("VIX", pd.DataFrame())
        oil_df = macro_data.get("CRUDE_OIL", pd.DataFrame())
        sp500_df = macro_data.get("SP500", pd.DataFrame())

        def get_1d_ret(df: pd.DataFrame) -> float:
            if not df.empty and len(df) > 1 and "close" in df.columns:
                ret = df["close"].pct_change().iloc[-1]
                return float(ret) if not pd.isna(ret) else 0.0
            return 0.0

        dxy_ret = get_1d_ret(dxy_df)
        oil_ret = get_1d_ret(oil_df)
        sp_ret = get_1d_ret(sp500_df)
        vix_val = float(vix_df.iloc[-1]["close"]) if not vix_df.empty and "close" in vix_df.columns else 18.0

        # Composite rules
        if vix_val > 25.0 or sp_ret < -0.015:
            sentiment = "Risk-Off"
        elif vix_val < 15.0 and sp_ret > 0.005:
            sentiment = "Risk-On"
        else:
            sentiment = "Neutral"

        if oil_ret > 0.02:
            inflation_stance = "Inflationary Pressure"
        elif oil_ret < -0.02:
            inflation_stance = "Disinflationary Pressure"
        else:
            inflation_stance = "Stable Inflation"

        if dxy_ret > 0.005:
            dollar_stance = "Dollar Strong"
        elif dxy_ret < -0.005:
            dollar_stance = "Dollar Weak"
        else:
            dollar_stance = "Dollar Neutral"

        composite_regime = f"{sentiment} | {inflation_stance} | {dollar_stance}"

        return {
            "macro_regime": composite_regime,
            "sentiment": sentiment,
            "inflation_stance": inflation_stance,
            "dollar_stance": dollar_stance,
            "vix": vix_val,
            "oil_1d_change": oil_ret,
            "dxy_1d_change": dxy_ret,
            "sp500_1d_change": sp_ret,
        }
