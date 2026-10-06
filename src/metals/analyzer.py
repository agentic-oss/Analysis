import logging
import pandas as pd
from typing import Dict, Any

from src.indicators.technical import TechnicalIndicators

logger = logging.getLogger(__name__)

class MetalsAnalyzer:
    """Dedicated analysis module for Gold and Silver price structure, support/resistance, and trend metrics."""

    @staticmethod
    def analyze_metal(df: pd.DataFrame, instrument_symbol: str) -> pd.DataFrame:
        if df.empty:
            return df
        # Calculate full technical indicator suite
        df = TechnicalIndicators.calculate_all(df)

        # Calculate Support & Resistance levels based on rolling 20d high/low
        df["support_20d"] = df["low"].rolling(window=20, min_periods=1).min()
        df["resistance_20d"] = df["high"].rolling(window=20, min_periods=1).max()
        df["pivot_point"] = (df["high"] + df["low"] + df["close"]) / 3.0

        return df

    @staticmethod
    def get_summary_metrics(df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            return {}
        last = df.iloc[-1]
        return {
            "date": last["date"],
            "price": float(last["close"]),
            "return_1d": float(last.get("return_1d", 0.0) or 0.0),
            "return_5d": float(last.get("return_5d", 0.0) or 0.0),
            "return_20d": float(last.get("return_20d", 0.0) or 0.0),
            "return_60d": float(last.get("return_60d", 0.0) or 0.0),
            "rsi_14": float(last.get("rsi_14", 50.0)),
            "macd": float(last.get("macd", 0.0)),
            "macd_hist": float(last.get("macd_hist", 0.0)),
            "atr_14": float(last.get("atr_14", 0.0)),
            "volatility_20d": float(last.get("volatility_20d", 0.0)),
            "dist_sma_200": float(last.get("dist_sma_200", 0.0)),
            "support_20d": float(last.get("support_20d", 0.0)),
            "resistance_20d": float(last.get("resistance_20d", 0.0))
        }
