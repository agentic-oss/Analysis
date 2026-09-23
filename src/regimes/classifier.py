import pandas as pd
import numpy as np
from typing import Dict, Any

class RegimeClassifier:
    """
    Transparent, explainable rule-based Macro and Market Regime Classification Engine.

    Macro Regimes:
    - Inflationary
    - Disinflationary / Deflationary
    - Risk-On
    - Risk-Off
    - Tightening
    - Easing
    - Stagflationary
    - Neutral

    Market Regimes (per metal):
    - Trend: Strong Bullish, Bullish, Range, Bearish, Strong Bearish
    - Volatility: High Volatility, Normal Volatility, Low Volatility
    """

    def classify_macro_regime(
        self,
        rates_change_60d: float = 0.0,
        dxy_return_60d: float = 0.0,
        cpi_yoy: float = 2.5,
        vix_level: float = 15.0,
        sp500_return_60d: float = 0.0,
        oil_return_60d: float = 0.0
    ) -> Dict[str, Any]:

        components = {
            "rates_trend": "tightening" if rates_change_60d > 0.2 else ("easing" if rates_change_60d < -0.2 else "neutral"),
            "dollar_trend": "strong" if dxy_return_60d > 2.0 else ("weak" if dxy_return_60d < -2.0 else "neutral"),
            "inflation_pressure": "high" if cpi_yoy > 3.5 or oil_return_60d > 10.0 else ("low" if cpi_yoy < 2.0 else "moderate"),
            "equity_sentiment": "risk_on" if sp500_return_60d > 3.0 and vix_level < 20.0 else ("risk_off" if vix_level > 22.0 or sp500_return_60d < -5.0 else "neutral")
        }

        # Rule-based primary macro regime decision
        if components["inflation_pressure"] == "high" and components["equity_sentiment"] == "risk_off":
            primary_regime = "Stagflationary"
        elif components["inflation_pressure"] == "high":
            primary_regime = "Inflationary"
        elif components["equity_sentiment"] == "risk_off":
            primary_regime = "Risk-Off"
        elif components["equity_sentiment"] == "risk_on":
            primary_regime = "Risk-On"
        elif components["rates_trend"] == "tightening":
            primary_regime = "Tightening"
        elif components["rates_trend"] == "easing":
            primary_regime = "Easing"
        else:
            primary_regime = "Neutral"

        return {
            "primary_macro_regime": primary_regime,
            "components": components,
            "inputs": {
                "rates_change_60d": rates_change_60d,
                "dxy_return_60d": dxy_return_60d,
                "cpi_yoy": cpi_yoy,
                "vix_level": vix_level,
                "sp500_return_60d": sp500_return_60d,
                "oil_return_60d": oil_return_60d
            }
        }

    def classify_market_regime(
        self,
        df_latest_row: pd.Series
    ) -> Dict[str, Any]:
        if df_latest_row.empty:
            return {"trend_regime": "Neutral", "volatility_regime": "Normal Volatility"}

        close = df_latest_row.get("close", 0.0)
        sma20 = df_latest_row.get("sma_20", close)
        sma50 = df_latest_row.get("sma_50", close)
        sma200 = df_latest_row.get("sma_200", close)
        rsi = df_latest_row.get("rsi_14", 50.0)
        vol_20d = df_latest_row.get("volatility_20d", 15.0)

        # Trend classification
        if close > sma20 > sma50 > sma200 and rsi > 55:
            trend = "Strong Bullish Trend"
        elif close > sma50 and sma50 > sma200:
            trend = "Bullish Trend"
        elif close < sma20 < sma50 < sma200 and rsi < 45:
            trend = "Strong Bearish Trend"
        elif close < sma50 and sma50 < sma200:
            trend = "Bearish Trend"
        else:
            trend = "Range"

        # Volatility classification
        if vol_20d > 22.0:
            vol_regime = "High Volatility"
        elif vol_20d < 10.0:
            vol_regime = "Low Volatility"
        else:
            vol_regime = "Normal Volatility"

        return {
            "trend_regime": trend,
            "volatility_regime": vol_regime,
            "rsi_14": round(float(rsi), 2) if not pd.isna(rsi) else 50.0,
            "volatility_20d": round(float(vol_20d), 2) if not pd.isna(vol_20d) else 15.0
        }
