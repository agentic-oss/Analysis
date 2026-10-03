import pandas as pd
import numpy as np
from typing import Dict, Any, List


class GoldAnalyzer:
    """Performs deep gold-specific analysis including rolling cross-asset correlations and support/resistance detection."""

    @staticmethod
    def calculate_rolling_correlations(
        gold_df: pd.DataFrame,
        macro_dfs: Dict[str, pd.DataFrame],
        windows: List[int] = [20, 60, 120, 252],
    ) -> Dict[str, Dict[int, float]]:
        """
        Calculate rolling correlations between Gold return and macro returns for multiple lookback windows.
        """
        results = {}

        if gold_df is None or gold_df.empty or "Close" not in gold_df.columns:
            return results

        gold_ret = gold_df["Close"].pct_change()
        gold_dates = pd.to_datetime(gold_df["Date"]).dt.date if "Date" in gold_df.columns else gold_df.index

        for asset_name, df in macro_dfs.items():
            if df is None or df.empty or "Close" not in df.columns:
                continue

            asset_ret = df["Close"].pct_change()
            asset_dates = pd.to_datetime(df["Date"]).dt.date if "Date" in df.columns else df.index

            merged = pd.DataFrame({"gold": gold_ret, "date": gold_dates}).merge(
                pd.DataFrame({"asset": asset_ret, "date": asset_dates}), on="date", how="inner"
            ).dropna()

            if len(merged) < 20:
                continue

            asset_corrs = {}
            for w in windows:
                if len(merged) >= w:
                    corr_val = merged["gold"].iloc[-w:].corr(merged["asset"].iloc[-w:])
                    asset_corrs[w] = round(float(corr_val), 4) if not np.isnan(corr_val) else 0.0
                else:
                    corr_val = merged["gold"].corr(merged["asset"])
                    asset_corrs[w] = round(float(corr_val), 4) if not np.isnan(corr_val) else 0.0

            results[asset_name] = asset_corrs

        return results

    @staticmethod
    def find_support_resistance(df: pd.DataFrame, window: int = 20) -> Dict[str, List[float]]:
        """Identify recent key support and resistance price levels."""
        if df is None or df.empty or len(df) < window:
            return {"support": [], "resistance": []}

        recent_lows = df["Low"].iloc[-window:].nsmallest(3).tolist()
        recent_highs = df["High"].iloc[-window:].nlargest(3).tolist()

        return {
            "support": [round(x, 2) for x in sorted(set(recent_lows))],
            "resistance": [round(x, 2) for x in sorted(set(recent_highs), reverse=True)],
        }
