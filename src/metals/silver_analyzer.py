import pandas as pd
import numpy as np
from typing import Dict, Any, List


class SilverAnalyzer:
    """Performs silver-specific technical and cross-market analysis."""

    @staticmethod
    def calculate_rolling_correlations(
        silver_df: pd.DataFrame,
        macro_dfs: Dict[str, pd.DataFrame],
        windows: List[int] = [20, 60, 120, 252],
    ) -> Dict[str, Dict[int, float]]:
        """Calculate rolling correlations between Silver returns and macro assets."""
        results = {}

        if silver_df is None or silver_df.empty or "Close" not in silver_df.columns:
            return results

        silver_ret = silver_df["Close"].pct_change()
        silver_dates = pd.to_datetime(silver_df["Date"]).dt.date if "Date" in silver_df.columns else silver_df.index

        for asset_name, df in macro_dfs.items():
            if df is None or df.empty or "Close" not in df.columns:
                continue

            asset_ret = df["Close"].pct_change()
            asset_dates = pd.to_datetime(df["Date"]).dt.date if "Date" in df.columns else df.index

            merged = pd.DataFrame({"silver": silver_ret, "date": silver_dates}).merge(
                pd.DataFrame({"asset": asset_ret, "date": asset_dates}), on="date", how="inner"
            ).dropna()

            if len(merged) < 20:
                continue

            asset_corrs = {}
            for w in windows:
                if len(merged) >= w:
                    corr_val = merged["silver"].iloc[-w:].corr(merged["asset"].iloc[-w:])
                    asset_corrs[w] = round(float(corr_val), 4) if not np.isnan(corr_val) else 0.0
                else:
                    corr_val = merged["silver"].corr(merged["asset"])
                    asset_corrs[w] = round(float(corr_val), 4) if not np.isnan(corr_val) else 0.0

            results[asset_name] = asset_corrs

        return results

    @staticmethod
    def find_support_resistance(df: pd.DataFrame, window: int = 20) -> Dict[str, List[float]]:
        """Identify recent key support and resistance price levels for silver."""
        if df is None or df.empty or len(df) < window:
            return {"support": [], "resistance": []}

        recent_lows = df["Low"].iloc[-window:].nsmallest(3).tolist()
        recent_highs = df["High"].iloc[-window:].nlargest(3).tolist()

        return {
            "support": [round(x, 2) for x in sorted(set(recent_lows))],
            "resistance": [round(x, 2) for x in sorted(set(recent_highs), reverse=True)],
        }
