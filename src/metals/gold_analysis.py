import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GoldAnalysis:
    """Analyzes Gold relationships with macro drivers (DXY, yields, real yields, oil, equities, VIX) and correlation regimes."""

    def __init__(self, windows: List[int] = [20, 60, 120, 252]):
        self.windows = windows

    def compute_cross_correlations(
        self, gold_df: pd.DataFrame, macro_df: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Computes rolling correlations between Gold returns and macro drivers:
        - DXY
        - US10Y
        - Real Yields / US2Y
        - Crude Oil
        - Equity (S&P 500 / NIFTY 50)
        - VIX
        """
        if gold_df.empty or macro_df.empty:
            return {}

        # Merge on date
        merged = pd.merge(
            gold_df[["date", "close"]].rename(columns={"close": "gold_close"}),
            macro_df,
            on="date",
            how="inner",
        ).sort_values("date")

        if len(merged) < 20:
            return {}

        # Calculate daily percentage returns / changes
        returns = pd.DataFrame({"date": merged["date"]})
        returns["gold_ret"] = merged["gold_close"].pct_change()

        macro_cols = [c for c in merged.columns if c not in ["date", "gold_close"]]
        for col in macro_cols:
            if "yield" in col.lower() or "10y" in col.lower() or "2y" in col.lower():
                returns[f"{col}_change"] = merged[col].diff()
            else:
                returns[f"{col}_ret"] = merged[col].pct_change()

        corr_results = {}
        for w in self.windows:
            w_corrs = pd.DataFrame({"date": returns["date"]})
            for col in returns.columns:
                if col not in ["date", "gold_ret"]:
                    w_corrs[f"gold_vs_{col}"] = returns["gold_ret"].rolling(window=w).corr(returns[col])
            corr_results[f"corr_{w}d"] = w_corrs

        return corr_results

    def analyze_gold_drivers(
        self, latest_gold_price: float, latest_correlations: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analyzes driver impact on Gold based on recent correlations."""
        summary = {
            "current_price": latest_gold_price,
            "correlations": latest_correlations,
            "dxy_decoupling": False,
            "real_yield_regime": "Normal",
        }

        dxy_corr_60 = latest_correlations.get("gold_vs_dxy_ret_60d", -0.5)
        if dxy_corr_60 is not None and abs(dxy_corr_60) < 0.2:
            summary["dxy_decoupling"] = True

        yield_corr_60 = latest_correlations.get("gold_vs_us10y_change_60d", -0.4)
        if yield_corr_60 is not None and yield_corr_60 > 0.1:
            summary["real_yield_regime"] = "Inflationary_Hedge"

        return summary
