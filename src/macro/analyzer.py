import pandas as pd
import numpy as np
from typing import Dict, Any, List

class MetalsMacroAnalyzer:
    """
    Analyzes cross-market relationships between precious metals (Gold, Silver)
    and macroeconomic drivers (DXY, Rates, Real Yields, Inflation, Energy, Equities, VIX).
    Calculates rolling correlations across multiple horizons (20d, 60d, 120d, 252d).
    """
    def __init__(self, windows: List[int] = [20, 60, 120, 252]):
        self.windows = windows

    def compute_cross_correlations(
        self,
        metal_df: pd.DataFrame,
        macro_dfs: Dict[str, pd.DataFrame],
        metal_col: str = "close"
    ) -> pd.DataFrame:
        """
        Calculates rolling correlations between metal returns and macro asset returns.
        """
        if metal_df.empty or "timestamp" not in metal_df.columns:
            return pd.DataFrame()

        base_df = metal_df[["timestamp", metal_col]].copy()
        base_df = base_df.sort_values("timestamp").reset_index(drop=True)
        base_df["metal_ret"] = base_df[metal_col].pct_change()

        merged = base_df[["timestamp", "metal_ret"]].copy()

        for macro_name, m_df in macro_dfs.items():
            if m_df.empty or "timestamp" not in m_df.columns or "close" not in m_df.columns:
                continue
            sub = m_df[["timestamp", "close"]].copy().sort_values("timestamp")
            sub[f"{macro_name}_ret"] = sub["close"].pct_change()
            merged = pd.merge(merged, sub[["timestamp", f"{macro_name}_ret"]], on="timestamp", how="inner")

        results = pd.DataFrame({"timestamp": merged["timestamp"]})

        for macro_name in macro_dfs.keys():
            col = f"{macro_name}_ret"
            if col in merged.columns:
                for w in self.windows:
                    corr_col = f"corr_{macro_name}_{w}d"
                    results[corr_col] = merged["metal_ret"].rolling(window=w, min_periods=max(5, w//4)).corr(merged[col])

        return results

    def get_latest_correlation_summary(
        self,
        metal_df: pd.DataFrame,
        macro_dfs: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, float]]:
        """
        Returns latest correlation values for each macro asset and window horizon.
        """
        corrs = self.compute_cross_correlations(metal_df, macro_dfs)
        if corrs.empty:
            return {}

        latest_row = corrs.iloc[-1]
        summary = {}

        for macro_name in macro_dfs.keys():
            summary[macro_name] = {}
            for w in self.windows:
                corr_col = f"corr_{macro_name}_{w}d"
                if corr_col in latest_row:
                    val = latest_row[corr_col]
                    summary[macro_name][f"{w}d"] = round(float(val), 4) if not pd.isna(val) else None

        return summary
