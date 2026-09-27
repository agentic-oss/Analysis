import pandas as pd
import numpy as np
from typing import Dict, Any, List
from src.indicators.technical import TechnicalIndicators
from src.macro.macro_analyzer import MacroAnalyzer

class GoldAnalyzer:
    """Multi-factor analysis module for Gold."""

    def __init__(self):
        self.macro_analyzer = MacroAnalyzer()

    def analyze(self, master_df: pd.DataFrame) -> pd.DataFrame:
        df = master_df.copy()
        if "GOLD_close" in df.columns:
            # Add technical indicators for Gold
            gold_df = pd.DataFrame({
                "close": df["GOLD_close"],
                "high": df["GOLD_high"] if "GOLD_high" in df.columns else df["GOLD_close"],
                "low": df["GOLD_low"] if "GOLD_low" in df.columns else df["GOLD_close"],
                "open": df["GOLD_open"] if "GOLD_open" in df.columns else df["GOLD_close"],
                "volume": df["GOLD_volume"] if "GOLD_volume" in df.columns else 0
            })
            gold_tech = TechnicalIndicators.calculate_all(gold_df)
            for col in gold_tech.columns:
                if col not in ["close", "high", "low", "open", "volume"]:
                    df[f"gold_{col}"] = gold_tech[col]

            # Cross-market correlations for Gold
            factors = [c for c in ["DXY_close", "US10Y_close", "REAL_YIELD_close", "CRUDE_OIL_close", "SP500_close", "VIX_close"] if c in df.columns]
            df = self.macro_analyzer.calculate_correlations(df, "GOLD_close", factors)

        return df

class SilverAnalyzer:
    """Multi-factor analysis module for Silver."""

    def __init__(self):
        self.macro_analyzer = MacroAnalyzer()

    def analyze(self, master_df: pd.DataFrame) -> pd.DataFrame:
        df = master_df.copy()
        if "SILVER_close" in df.columns:
            # Add technical indicators for Silver
            silver_df = pd.DataFrame({
                "close": df["SILVER_close"],
                "high": df["SILVER_high"] if "SILVER_high" in df.columns else df["SILVER_close"],
                "low": df["SILVER_low"] if "SILVER_low" in df.columns else df["SILVER_close"],
                "open": df["SILVER_open"] if "SILVER_open" in df.columns else df["SILVER_close"],
                "volume": df["SILVER_volume"] if "SILVER_volume" in df.columns else 0
            })
            silver_tech = TechnicalIndicators.calculate_all(silver_df)
            for col in silver_tech.columns:
                if col not in ["close", "high", "low", "open", "volume"]:
                    df[f"silver_{col}"] = silver_tech[col]

            # Cross-market correlations for Silver
            factors = [c for c in ["DXY_close", "US10Y_close", "REAL_YIELD_close", "CRUDE_OIL_close", "SP500_close", "VIX_close", "GOLD_close"] if c in df.columns]
            df = self.macro_analyzer.calculate_correlations(df, "SILVER_close", factors)

        return df
