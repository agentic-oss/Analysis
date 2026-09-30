import logging
from typing import Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)


class MarketRegimeClassifier:
    """Classifies asset-specific trend and volatility market regimes."""

    @staticmethod
    def classify_market_regime(
        price: float,
        sma_20: float,
        sma_50: float,
        sma_200: float,
        volatility_20d: float,
        rsi_14: float,
    ) -> Dict[str, Any]:
        """
        Classifies asset market regime across dimensions:
        - Trend: Strong Bullish Trend, Bullish Trend, Range, Bearish Trend, Strong Bearish Trend
        - Volatility: High Volatility, Low Volatility, Normal Volatility
        """
        # Trend classification
        if price > sma_20 > sma_50 > sma_200:
            trend_regime = "Strong Bullish Trend"
        elif price > sma_50 and sma_50 > sma_200:
            trend_regime = "Bullish Trend"
        elif price < sma_20 < sma_50 < sma_200:
            trend_regime = "Strong Bearish Trend"
        elif price < sma_50 and sma_50 < sma_200:
            trend_regime = "Bearish Trend"
        else:
            trend_regime = "Range"

        # Volatility regime
        if volatility_20d > 0.25:
            volatility_regime = "High Volatility"
        elif volatility_20d < 0.12:
            volatility_regime = "Low Volatility"
        else:
            volatility_regime = "Normal Volatility"

        return {
            "trend_regime": trend_regime,
            "volatility_regime": volatility_regime,
            "combined_regime": f"{trend_regime} | {volatility_regime}",
            "rsi_state": "Overbought" if rsi_14 > 70 else ("Oversold" if rsi_14 < 30 else "Neutral"),
        }
