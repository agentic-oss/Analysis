"""
Gold analysis module calculating multi-factor rolling correlations and key market relationships.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


class GoldAnalysis:
    """Analyzes Gold relationships with macro drivers (DXY, Yields, Oil, Equities, VIX)."""

    def __init__(self, windows: list = [20, 60, 120, 252]):
        self.windows = windows

    def calculate_rolling_correlations(self, gold_df: pd.DataFrame, macro_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculates rolling correlations between Gold returns and key macro series returns.
        macro_dfs dict contains keys like: 'dxy', 'us10y', 'oil', 'sp500', 'vix'.
        """
        if gold_df.empty or "close" not in gold_df.columns:
            return pd.DataFrame()

        df = pd.DataFrame({"date": gold_df["date"], "gold_ret": gold_df["close"].pct_change()})

        for key, m_df in macro_dfs.items():
            if not m_df.empty and "close" in m_df.columns and "date" in m_df.columns:
                m_ret = m_df.set_index("date")["close"].pct_change().rename(f"{key}_ret")
                df = df.merge(m_ret, on="date", how="left")

        df = df.set_index("date")

        res_df = pd.DataFrame(index=df.index)

        for col in df.columns:
            if col == "gold_ret":
                continue
            key_name = col.replace("_ret", "")
            for w in self.windows:
                corr_col = f"corr_gold_{key_name}_{w}d"
                res_df[corr_col] = df["gold_ret"].rolling(window=w, min_periods=min(10, w)).corr(df[col])

        return res_df.reset_index()
