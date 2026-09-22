"""
Market regime detection for individual instruments (trend, volatility, and combined market states).
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


class MarketRegimeDetector:
    """
    Classifies instrument-specific market regime into multidimensional components:
    - Trend Regime: Strong Bullish Trend, Bullish Trend, Range, Bearish Trend, Strong Bearish Trend
    - Volatility Regime: High Volatility, Medium Volatility, Low Volatility
    """

    @staticmethod
    def classify_instrument(df: pd.DataFrame) -> pd.DataFrame:
        """
        Classifies trend and volatility regimes for a given instrument OHLCV dataframe with indicators.
        Requires columns: 'close', 'sma_20', 'sma_50', 'sma_200', 'rsi_14', 'volatility_20d'.
        """
        if df.empty or "close" not in df.columns:
            return df

        df = df.copy()

        trend_regimes = []
        vol_regimes = []

        # Volatility quantile threshold calculation if sufficient rows, else static thresholds
        if len(df) >= 60 and "volatility_20d" in df.columns:
            vol_q33 = df["volatility_20d"].quantile(0.33)
            vol_q66 = df["volatility_20d"].quantile(0.66)
        else:
            vol_q33, vol_q66 = 0.12, 0.22

        for idx, row in df.iterrows():
            close = row.get("close", np.nan)
            sma_20 = row.get("sma_20", close)
            sma_50 = row.get("sma_50", close)
            sma_200 = row.get("sma_200", close)
            rsi = row.get("rsi_14", 50.0)
            vol = row.get("volatility_20d", 0.15)

            # Trend Classification
            if close > sma_20 > sma_50 > sma_200 and rsi > 55:
                trend = "Strong Bullish Trend"
            elif close > sma_50 > sma_200:
                trend = "Bullish Trend"
            elif close < sma_20 < sma_50 < sma_200 and rsi < 45:
                trend = "Strong Bearish Trend"
            elif close < sma_50 < sma_200:
                trend = "Bearish Trend"
            else:
                trend = "Range"

            # Volatility Classification
            if vol >= vol_q66:
                vol_regime = "High Volatility"
            elif vol <= vol_q33:
                vol_regime = "Low Volatility"
            else:
                vol_regime = "Medium Volatility"

            trend_regimes.append(trend)
            vol_regimes.append(vol_regime)

        df["trend_regime"] = trend_regimes
        df["volatility_regime"] = vol_regimes
        df["market_regime"] = df["trend_regime"] + " / " + df["volatility_regime"]

        return df
