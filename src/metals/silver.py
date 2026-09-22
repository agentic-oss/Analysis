"""
Silver analysis module calculating multi-factor relationships and industrial driver correlations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


class SilverAnalysis:
    """Analyzes Silver relationships with gold, commodities, and macro variables."""

    def __init__(self, windows: list = [20, 60, 120, 252]):
        self.windows = windows

    def calculate_rolling_correlations(self, silver_df: pd.DataFrame, macro_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculates rolling correlations for Silver returns against macro/commodity series."""
        if silver_df.empty or "close" not in silver_df.columns:
            return pd.DataFrame()

        df = pd.DataFrame({"date": silver_df["date"], "silver_ret": silver_df["close"].pct_change()})

        for key, m_df in macro_dfs.items():
            if not m_df.empty and "close" in m_df.columns and "date" in m_df.columns:
                m_ret = m_df.set_index("date")["close"].pct_change().rename(f"{key}_ret")
                df = df.merge(m_ret, on="date", how="left")

        df = df.set_index("date")
        res_df = pd.DataFrame(index=df.index)

        for col in df.columns:
            if col == "silver_ret":
                continue
            key_name = col.replace("_ret", "")
            for w in self.windows:
                corr_col = f"corr_silver_{key_name}_{w}d"
                res_df[corr_col] = df["silver_ret"].rolling(window=w, min_periods=min(10, w)).corr(df[col])

        return res_df.reset_index()
