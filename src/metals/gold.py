"""
Gold Analysis Module.
Analyzes cross-asset rolling correlations (20, 60, 120, 252 days) between Gold and:
DXY, US 10Y Yield, Real Yields, Inflation, Oil, S&P 500, NIFTY 50, VIX.
Detects correlation regime shifts.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


class GoldAnalyzer:
    """Performs deep cross-market correlation and macro driver analysis for Gold."""

    def __init__(self, windows: List[int] = [20, 60, 120, 252]):
        self.windows = windows

    def calculate_cross_correlations(
        self,
        gold_df: pd.DataFrame,
        macro_dfs: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Calculates rolling correlation matrix of Gold daily returns vs macro assets across windows.
        Returns a DataFrame indexed by date with rolling correlation features.
        """
        if gold_df.empty or "close" not in gold_df.columns:
            return pd.DataFrame()

        merged = gold_df[["date", "close"]].rename(columns={"close": "GOLD_close"}).copy()
        merged["GOLD_ret"] = merged["GOLD_close"].pct_change()

        for asset, df in macro_dfs.items():
            if df.empty or "close" not in df.columns:
                continue
            m = df[["date", "close"]].rename(columns={"close": f"{asset}_close"})
            merged = pd.merge(merged, m, on="date", how="left")
            merged[f"{asset}_ret"] = merged[f"{asset}_close"].pct_change()

        # Compute rolling correlations for each asset and window
        res_df = pd.DataFrame({"date": merged["date"]})
        asset_keys = [k for k in macro_dfs.keys() if f"{k}_ret" in merged.columns]

        for asset in asset_keys:
            for w in self.windows:
                res_df[f"corr_gold_{asset.lower()}_{w}d"] = (
                    merged["GOLD_ret"].rolling(window=w, min_periods=min(10, w)).corr(merged[f"{asset}_ret"])
                )

        return res_df

    def detect_regime_shifts(self, corr_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detects significant shifts in 60d rolling correlations over the last 20 trading days."""
        shifts = []
        if corr_df.empty or len(corr_df) < 20:
            return shifts

        curr = corr_df.iloc[-1]
        past = corr_df.iloc[-20]

        for col in corr_df.columns:
            if col.startswith("corr_gold_") and col.endswith("_60d"):
                c_now = curr[col]
                c_past = past[col]
                if pd.notna(c_now) and pd.notna(c_past):
                    diff = c_now - c_past
                    if abs(diff) >= 0.35: # Threshold for major correlation shift
                        shifts.append({
                            "feature": col,
                            "current_corr": float(c_now),
                            "past_corr": float(c_past),
                            "shift": float(diff),
                            "description": f"Significant correlation shift in {col}: {c_past:.2f} -> {c_now:.2f}"
                        })
        return shifts
