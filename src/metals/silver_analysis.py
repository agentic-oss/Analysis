import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SilverAnalysis:
    """Analyzes Silver specific relationships including Silver vs Gold, Copper, Equities, and DXY."""

    @staticmethod
    def analyze_silver_dynamics(
        silver_df: pd.DataFrame, gold_df: pd.DataFrame, dxy_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Calculates relative performance of Silver vs Gold and correlation statistics."""
        if silver_df.empty or gold_df.empty:
            return {}

        merged = pd.merge(
            silver_df[["date", "close"]].rename(columns={"close": "silver_close"}),
            gold_df[["date", "close"]].rename(columns={"close": "gold_close"}),
            on="date",
            how="inner",
        ).sort_values("date")

        if len(merged) < 20:
            return {}

        merged["silver_ret_1d"] = merged["silver_close"].pct_change()
        merged["gold_ret_1d"] = merged["gold_close"].pct_change()
        merged["silver_gold_spread"] = merged["silver_ret_1d"] - merged["gold_ret_1d"]

        # 20-day rolling beta of Silver to Gold
        cov = merged["silver_ret_1d"].rolling(20).cov(merged["gold_ret_1d"])
        var = merged["gold_ret_1d"].rolling(20).var()
        merged["silver_gold_beta"] = cov / var.replace(0, np.nan)

        latest = merged.iloc[-1]
        return {
            "silver_close": float(latest["silver_close"]),
            "gold_close": float(latest["gold_close"]),
            "silver_gold_beta_20d": float(latest["silver_gold_beta"]) if pd.notnull(latest["silver_gold_beta"]) else None,
            "recent_spread_1d": float(latest["silver_gold_spread"]) if pd.notnull(latest["silver_gold_spread"]) else None,
        }
