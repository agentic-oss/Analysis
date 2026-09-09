"""
Macro & Market Regime Classification Module.
Defines transparent rule-based composite frameworks for Macro regimes
(Inflationary, Disinflationary, Deflationary, Risk-On, Risk-Off, Tightening, Easing, Stagflationary, Neutral)
and Market regimes for Gold and Silver (Strong Bullish, Bullish, Range, Bearish, Strong Bearish, High/Low Volatility).
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


class RegimeClassifier:
    """Classifies global macro state and individual asset market regimes."""

    def classify_macro_regime(
        self,
        rates_change_20d: float = 0.0,
        real_yield_change_20d: float = 0.0,
        dxy_return_20d: float = 0.0,
        cpi_surprise: float = 0.0,
        vix_level: float = 18.0,
        oil_return_20d: float = 0.0,
        equity_return_20d: float = 0.0
    ) -> Dict[str, Any]:
        """
        Transparent rule-based composite macro regime detector.
        Returns regime name along with underlying component scores for complete explainability.
        """
        # Component signals
        inflation_signal = "HIGH" if (cpi_surprise > 0 or oil_return_20d > 5.0) else ("LOW" if oil_return_20d < -5.0 else "NEUTRAL")
        policy_signal = "TIGHTENING" if rates_change_20d > 0.1 else ("EASING" if rates_change_20d < -0.1 else "NEUTRAL")
        risk_signal = "RISK_OFF" if (vix_level > 22.0 or equity_return_20d < -3.0) else ("RISK_ON" if equity_return_20d > 2.0 else "NEUTRAL")

        regime = "NEUTRAL"
        if inflation_signal == "HIGH" and equity_return_20d < -2.0:
            regime = "STAGFLATIONARY"
        elif inflation_signal == "HIGH":
            regime = "INFLATIONARY"
        elif risk_signal == "RISK_OFF":
            regime = "RISK_OFF"
        elif risk_signal == "RISK_ON":
            regime = "RISK_ON"
        elif policy_signal == "TIGHTENING":
            regime = "TIGHTENING"
        elif policy_signal == "EASING":
            regime = "EASING"
        elif oil_return_20d < -8.0 and rates_change_20d < -0.1:
            regime = "DEFLATIONARY"

        return {
            "macro_regime": regime,
            "components": {
                "inflation_signal": inflation_signal,
                "policy_signal": policy_signal,
                "risk_signal": risk_signal,
                "rates_change_20d": rates_change_20d,
                "vix_level": vix_level,
                "oil_return_20d": oil_return_20d,
                "equity_return_20d": equity_return_20d
            }
        }

    def classify_market_regime(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Classifies price trend and volatility regime for an individual metal (Gold or Silver).
        Supported Trend Regimes: Strong Bullish, Bullish, Range, Bearish, Strong Bearish
        Supported Volatility Regimes: High Volatility, Low Volatility, Normal Volatility
        """
        if df.empty or "close" not in df.columns or "sma_50" not in df.columns:
            return {"trend_regime": "NEUTRAL", "volatility_regime": "NORMAL"}

        curr = df.iloc[-1]
        close = curr["close"]
        sma20 = curr.get("sma_20", close)
        sma50 = curr.get("sma_50", close)
        sma200 = curr.get("sma_200", close)
        vol20 = curr.get("volatility_20d", 15.0)

        # Trend classification
        if close > sma20 and sma20 > sma50 and sma50 > sma200:
            trend = "STRONG_BULLISH"
        elif close > sma50 and sma50 > sma200:
            trend = "BULLISH"
        elif close < sma20 and sma20 < sma50 and sma50 < sma200:
            trend = "STRONG_BEARISH"
        elif close < sma50 and sma50 < sma200:
            trend = "BEARISH"
        else:
            trend = "RANGE"

        # Volatility classification
        if vol20 > 22.0:
            vol_regime = "HIGH_VOLATILITY"
        elif vol20 < 10.0:
            vol_regime = "LOW_VOLATILITY"
        else:
            vol_regime = "NORMAL_VOLATILITY"

        return {
            "trend_regime": trend,
            "volatility_regime": vol_regime,
            "close": float(close),
            "sma20": float(sma20),
            "sma50": float(sma50),
            "sma200": float(sma200),
            "volatility_20d": float(vol20)
        }
