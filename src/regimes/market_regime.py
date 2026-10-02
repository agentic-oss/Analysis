import pandas as pd
import numpy as np

class MarketRegimeClassifier:
    """Classifies asset market regimes across Trend and Volatility dimensions."""

    def classify_market_regime(self, df: pd.DataFrame) -> dict:
        """
        Calculates trend and volatility regime for a metal given technical indicators dataframe.
        Expects last row to contain 'close', 'sma_20', 'sma_50', 'sma_200', 'dist_sma_200', 'volatility_20d', 'rsi_14'.
        """
        if df.empty:
            return {"trend_regime": "Unknown", "volatility_regime": "Unknown"}

        latest = df.iloc[-1]

        close = latest.get("close", 0.0)
        sma_20 = latest.get("sma_20", close)
        sma_50 = latest.get("sma_50", close)
        sma_200 = latest.get("sma_200", close)
        vol_20d = latest.get("volatility_20d", 0.15) or 0.15
        rsi = latest.get("rsi_14", 50.0) or 50.0

        # Trend classification
        if close > sma_20 and sma_20 > sma_50 and sma_50 > sma_200:
            trend_regime = "Strong Bullish Trend"
        elif close > sma_50 and sma_50 > sma_200:
            trend_regime = "Bullish Trend"
        elif close < sma_20 and sma_20 < sma_50 and sma_50 < sma_200:
            trend_regime = "Strong Bearish Trend"
        elif close < sma_50 and sma_50 < sma_200:
            trend_regime = "Bearish Trend"
        else:
            trend_regime = "Range / Consolidation"

        # Volatility classification
        hist_vol_avg = df["volatility_20d"].mean() if "volatility_20d" in df.columns else 0.15
        if vol_20d > hist_vol_avg * 1.3:
            vol_regime = "High Volatility"
        elif vol_20d < hist_vol_avg * 0.7:
            vol_regime = "Low Volatility"
        else:
            vol_regime = "Normal Volatility"

        # Composite description
        return {
            "trend_regime": trend_regime,
            "volatility_regime": vol_regime,
            "rsi_status": "Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Neutral"),
            "composite_label": f"{trend_regime} ({vol_regime})"
        }
