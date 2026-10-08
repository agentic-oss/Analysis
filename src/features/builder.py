"""
Daily Feature Store Builder for Precious Metals & Macro Data.
Constructs ML-ready datasets and target variables without look-ahead bias.
"""
from typing import Dict, List, Any
import pandas as pd
import numpy as np


class FeatureBuilder:
    """
    Builds aligned daily feature tables and target variables.
    """

    @staticmethod
    def build_daily_features(
        metal_df: pd.DataFrame,
        symbol: str,
        macro_dfs: Dict[str, pd.DataFrame] = None,
        ratio_df: pd.DataFrame = None,
        horizons: List[int] = [1, 3, 5, 10, 20, 60],
    ) -> pd.DataFrame:
        """
        Builds feature dataset for symbol from technical indicators, macro drivers, and relative value ratios.
        """
        if metal_df.empty:
            return pd.DataFrame()

        df = metal_df.copy().sort_values("date").reset_index(drop=True)
        df["instrument"] = symbol

        # Basic returns
        df["return_1d"] = df["close"].pct_change(1)
        df["return_5d"] = df["close"].pct_change(5)
        df["return_20d"] = df["close"].pct_change(20)

        # Merge ratio features if available
        if ratio_df is not None and not ratio_df.empty:
            r_cols = ratio_df[["date", "gold_silver_ratio", "ratio_zscore", "spread_return_1d"]]
            df = pd.merge(df, r_cols, on="date", how="left")

        # Merge macro features if available
        if macro_dfs:
            for m_sym, m_df in macro_dfs.items():
                if not m_df.empty and "close" in m_df.columns:
                    m_sub = m_df[["date", "close"]].rename(columns={"close": f"{m_sym.lower()}_close"})
                    m_sub[f"{m_sym.lower()}_return_1d"] = m_sub[f"{m_sym.lower()}_close"].pct_change(1)
                    df = pd.merge(df, m_sub[[col for col in m_sub.columns if col != f"{m_sym.lower()}_close"]], on="date", how="left")

        # Target variables (Forward returns - strictly shifted backwards from the future)
        for h in horizons:
            # future price at t+h / price at t - 1
            df[f"future_return_{h}d"] = (df["close"].shift(-h) / df["close"]) - 1.0
            df[f"future_direction_{h}d"] = (df[f"future_return_{h}d"] > 0).astype(int)

        return df
