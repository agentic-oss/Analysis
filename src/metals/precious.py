"""
Dedicated Gold and Silver Analysis Modules with Cross-Asset Correlation Engines.
Computes 20, 60, 120, and 252-day rolling correlations against DXY, yields, oil, equities, VIX, USD/INR.
"""
import logging
from typing import Dict, List, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CrossAssetCorrelationEngine:
    """
    Computes rolling multi-window cross-asset correlations between precious metals and macro drivers.
    Windows: 20, 60, 120, 252 days.
    """

    def __init__(self, windows: List[int] = None):
        self.windows = windows or [20, 60, 120, 252]

    def compute_cross_correlations(
        self,
        metal_df: pd.DataFrame,
        macro_dfs: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Aligns metal price series with macro price series on 'date' and computes rolling return correlations.
        `macro_dfs` map asset_key -> DataFrame containing ['date', 'close'].
        """
        if metal_df.empty or "close" not in metal_df.columns:
            return metal_df

        df_out = metal_df.copy().sort_values("date").reset_index(drop=True)
        df_out["metal_return"] = df_out["close"].pct_change()

        # Merge macro closes
        merged = df_out[["date", "metal_return"]].copy()

        for asset_key, m_df in macro_dfs.items():
            if m_df.empty or "close" not in m_df.columns:
                continue
            m_sub = m_df[["date", "close"]].rename(columns={"close": f"close_{asset_key}"})
            m_sub[f"return_{asset_key}"] = m_sub[f"close_{asset_key}"].pct_change()
            merged = pd.merge(merged, m_sub[["date", f"return_{asset_key}"]], on="date", how="left")

        # Compute rolling correlations
        for asset_key in macro_dfs.keys():
            col_name = f"return_{asset_key}"
            if col_name not in merged.columns:
                continue
            for win in self.windows:
                corr_series = merged["metal_return"].rolling(window=win, min_periods=max(5, win // 4)).corr(merged[col_name])
                df_out[f"corr_{asset_key}_{win}d"] = corr_series

        return df_out


class GoldAnalyzer:
    """
    Gold specific multi-factor analysis module.
    """

    def __init__(self, corr_engine: CrossAssetCorrelationEngine = None):
        self.corr_engine = corr_engine or CrossAssetCorrelationEngine()

    def analyze(self, gold_df: pd.DataFrame, macro_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        df_analyzed = self.corr_engine.compute_cross_correlations(gold_df, macro_dfs)
        return df_analyzed


class SilverAnalyzer:
    """
    Silver specific multi-factor analysis module.
    """

    def __init__(self, corr_engine: CrossAssetCorrelationEngine = None):
        self.corr_engine = corr_engine or CrossAssetCorrelationEngine()

    def analyze(self, silver_df: pd.DataFrame, macro_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        df_analyzed = self.corr_engine.compute_cross_correlations(silver_df, macro_dfs)
        return df_analyzed
