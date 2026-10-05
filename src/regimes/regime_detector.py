"""Rule-based Macro Regime and Market Regime Detection."""

from typing import Dict, Any
import pandas as pd


class RegimeDetector:
    """Classifies global Macro Regime and asset-specific Market Regimes."""

    def detect_macro_regime(
        self,
        rates_change_20d: float,
        dxy_change_20d: float,
        vix_level: float,
        oil_change_20d: float,
        sp500_change_20d: float
    ) -> Dict[str, Any]:
        """
        Determines transparent macro regime from multi-factor conditions.
        Regimes: Inflationary, Disinflationary, Risk-On, Risk-Off, Tightening, Easing, Neutral.
        """
        components = {
            "rates_trend": "rising" if rates_change_20d > 0.02 else ("falling" if rates_change_20d < -0.02 else "flat"),
            "dxy_trend": "strong" if dxy_change_20d > 0.01 else ("weak" if dxy_change_20d < -0.01 else "flat"),
            "vix_state": "elevated" if vix_level > 20 else "calm",
            "oil_trend": "rising" if oil_change_20d > 0.03 else ("falling" if oil_change_20d < -0.03 else "flat"),
            "equity_trend": "bullish" if sp500_change_20d > 0.01 else ("bearish" if sp500_change_20d < -0.01 else "flat")
        }

        # Rule evaluation
        if vix_level > 25 or (sp500_change_20d < -0.03 and dxy_change_20d > 0.01):
            primary_regime = "Risk-Off"
        elif oil_change_20d > 0.05 and rates_change_20d > 0.02:
            primary_regime = "Inflationary"
        elif rates_change_20d > 0.03 and dxy_change_20d > 0.02:
            primary_regime = "Tightening"
        elif sp500_change_20d > 0.02 and vix_level < 18:
            primary_regime = "Risk-On"
        elif rates_change_20d < -0.03 and dxy_change_20d < -0.02:
            primary_regime = "Easing"
        else:
            primary_regime = "Neutral"

        return {
            "primary_regime": primary_regime,
            "explainable_components": components
        }

    def detect_market_regime(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Detects specific market regime for Gold or Silver:
        Trend: Strong Bullish Trend, Bullish Trend, Range, Bearish Trend, Strong Bearish Trend
        Volatility: High Volatility, Low Volatility
        """
        if df.empty or 'close' not in df.columns:
            return {"trend_regime": "Neutral", "volatility_regime": "Normal"}

        latest = df.iloc[-1]
        close = latest.get('close', 0.0)
        sma_20 = latest.get('sma_20', close)
        sma_50 = latest.get('sma_50', close)
        sma_200 = latest.get('sma_200', close)
        vol_20d = latest.get('volatility_20d', 0.15)

        # Trend regime
        if close > sma_20 > sma_50 > sma_200:
            trend = "Strong Bullish Trend"
        elif close > sma_50:
            trend = "Bullish Trend"
        elif close < sma_20 < sma_50 < sma_200:
            trend = "Strong Bearish Trend"
        elif close < sma_50:
            trend = "Bearish Trend"
        else:
            trend = "Range"

        # Volatility regime
        if vol_20d > 0.22:
            vol_regime = "High Volatility"
        elif vol_20d < 0.10:
            vol_regime = "Low Volatility"
        else:
            vol_regime = "Normal Volatility"

        return {
            "trend_regime": trend,
            "volatility_regime": vol_regime,
            "volatility_20d_annualized": round(float(vol_20d) if pd.notna(vol_20d) else 0.15, 4)
        }
