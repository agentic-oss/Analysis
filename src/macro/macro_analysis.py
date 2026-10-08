"""
Macro Cross-Market Correlation and Driver Analysis.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any


class MacroAnalyzer:
    """
    Analyzes cross-market relationships and rolling correlations between precious metals and macro drivers (DXY, Yields, Oil, Equities, VIX).
    """

    @staticmethod
    def calculate_rolling_correlations(
        metal_df: pd.DataFrame,
        macro_dfs: Dict[str, pd.DataFrame],
        windows: List[int] = [20, 60, 120, 252],
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculates latest rolling correlations between metal close prices/returns and macro drivers across multiple windows.
        """
        if metal_df.empty:
            return {}

        metal_ret = metal_df.set_index("date")["close"].pct_change().dropna()
        results = {}

        for macro_name, macro_df in macro_dfs.items():
            if macro_df.empty or "close" not in macro_df.columns:
                continue

            m_ret = macro_df.set_index("date")["close"].pct_change().dropna()
            aligned = pd.concat([metal_ret, m_ret], axis=1, keys=["metal", "macro"]).dropna()

            if len(aligned) < 20:
                continue

            results[macro_name] = {}
            for w in windows:
                if len(aligned) >= w:
                    corr = float(aligned["metal"].tail(w).corr(aligned["macro"].tail(w)))
                    results[macro_name][f"corr_{w}d"] = corr if not np.isnan(corr) else 0.0
                else:
                    results[macro_name][f"corr_{w}d"] = float(aligned["metal"].corr(aligned["macro"]))

        return results
