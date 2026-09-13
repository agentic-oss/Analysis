import logging
import pandas as pd
from typing import Dict, Any

logger = logging.getLogger(__name__)

class MacroRegimeClassifier:
    """Transparent rule-based macro regime classification incorporating rates, yields, inflation, DXY, VIX, Oil, Equities."""

    @staticmethod
    def classify_macro_regime(
        us10y_change_20d: float,
        real_yield_change_20d: float,
        dxy_return_20d: float,
        vix_level: float,
        oil_return_20d: float,
        equity_return_20d: float
    ) -> Dict[str, Any]:
        """
        Classifies macro regime into one of:
        Inflationary, Disinflationary, Deflationary, Risk-On, Risk-Off, Tightening, Easing, Stagflationary, Neutral.
        """
        components = {
            "us10y_change_20d": us10y_change_20d,
            "real_yield_change_20d": real_yield_change_20d,
            "dxy_return_20d": dxy_return_20d,
            "vix_level": vix_level,
            "oil_return_20d": oil_return_20d,
            "equity_return_20d": equity_return_20d
        }

        # Multi-factor rules
        if vix_level > 25.0 and equity_return_20d < -0.04:
            regime = "Risk-Off"
        elif oil_return_20d > 0.08 and us10y_change_20d > 0.15 and equity_return_20d < -0.02:
            regime = "Stagflationary"
        elif oil_return_20d > 0.05 and us10y_change_20d > 0.10:
            regime = "Inflationary"
        elif real_yield_change_20d > 0.20 and dxy_return_20d > 0.02:
            regime = "Tightening"
        elif real_yield_change_20d < -0.20 and dxy_return_20d < -0.02:
            regime = "Easing"
        elif equity_return_20d > 0.03 and vix_level < 18.0:
            regime = "Risk-On"
        elif oil_return_20d < -0.08 and equity_return_20d < -0.03:
            regime = "Deflationary"
        elif us10y_change_20d < -0.10 and oil_return_20d < -0.03:
            regime = "Disinflationary"
        else:
            regime = "Neutral"

        return {
            "macro_regime": regime,
            "explainable_components": components
        }


class MarketRegimeClassifier:
    """Classifies instrument-specific trend and volatility regimes simultaneously."""

    @staticmethod
    def classify_market_regime(
        price: float,
        sma_50: float,
        sma_200: float,
        rsi_14: float,
        hist_vol_20: float
    ) -> Dict[str, Any]:
        # Trend
        if price > sma_50 > sma_200 and rsi_14 > 55:
            trend = "Strong Bullish Trend"
        elif price > sma_50 or price > sma_200:
            trend = "Bullish Trend"
        elif price < sma_50 < sma_200 and rsi_14 < 45:
            trend = "Strong Bearish Trend"
        elif price < sma_50 or price < sma_200:
            trend = "Bearish Trend"
        else:
            trend = "Range"

        # Volatility
        vol_regime = "High Volatility" if hist_vol_20 > 22.0 else "Low Volatility"

        return {
            "trend_regime": trend,
            "volatility_regime": vol_regime,
            "composite_market_regime": f"{trend} | {vol_regime}"
        }
