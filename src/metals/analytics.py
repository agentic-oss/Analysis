import logging
from typing import Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class MetalCrossMarketAnalytics:
    """Calculates cross-market rolling correlations and relationships for Gold & Silver."""

    @staticmethod
    def calculate_rolling_correlations(
        metal_df: pd.DataFrame,
        macro_datasets: Dict[str, pd.DataFrame],
        windows: List[int] = [20, 60, 120, 252],
    ) -> pd.DataFrame:
        """Calculates rolling correlations between metal returns and macro series."""
        if metal_df.empty or "close" not in metal_df.columns:
            return metal_df

        res = metal_df.copy().sort_values("date").reset_index(drop=True)
        res["metal_return"] = res["close"].pct_change()

        macro_returns = {}
        for sym, mdf in macro_datasets.items():
            if not mdf.empty and "close" in mdf.columns and "date" in mdf.columns:
                m = mdf[["date", "close"]].copy().sort_values("date").reset_index(drop=True)
                m[f"{sym.lower()}_return"] = m["close"].pct_change()
                macro_returns[sym] = m[["date", f"{sym.lower()}_return"]]

        # Merge returns by date
        for sym, m_df in macro_returns.items():
            res = pd.merge(res, m_df, on="date", how="left")
            for w in windows:
                corr_col = f"corr_{sym.lower()}_{w}d"
                res[corr_col] = (
                    res["metal_return"]
                    .rolling(w, min_periods=max(5, w // 4))
                    .corr(res[f"{sym.lower()}_return"])
                )

        return res
