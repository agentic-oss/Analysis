import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class MacroCorrelationAnalyzer:
    """Computes rolling multi-window correlations between metals and macro drivers (DXY, Real Yields, 10Y, Oil, VIX, Equities)."""

    @staticmethod
    def compute_correlations(metal_df: pd.DataFrame, macro_dfs: Dict[str, pd.DataFrame], windows: List[int] = [20, 60, 120, 252]) -> pd.DataFrame:
        if metal_df.empty or not macro_dfs:
            return pd.DataFrame()

        base = metal_df[["date", "close"]].rename(columns={"close": "metal_price"}).copy()
        base["metal_return"] = base["metal_price"].pct_change(1)

        for key, df in macro_dfs.items():
            if df is not None and not df.empty and "close" in df.columns:
                m_sub = df[["date", "close"]].rename(columns={"close": f"{key}_price"}).copy()
                m_sub[f"{key}_return"] = m_sub[f"{key}_price"].pct_change(1)
                base = pd.merge(base, m_sub[["date", f"{key}_return"]], on="date", how="left")

        # Rolling correlations
        for key in macro_dfs.keys():
            col_ret = f"{key}_return"
            if col_ret in base.columns:
                for w in windows:
                    base[f"corr_{key}_{w}d"] = base["metal_return"].rolling(window=w, min_periods=min(10, w)).corr(base[col_ret])

        return base

    @staticmethod
    def detect_regime_shifts(corr_df: pd.DataFrame, macro_key: str, window: int = 60, threshold: float = 0.5) -> List[Dict[str, Any]]:
        shifts = []
        col = f"corr_{macro_key}_{window}d"
        if corr_df.empty or col not in corr_df.columns:
            return shifts

        s = corr_df[col].dropna()
        if len(s) < 20:
            return shifts

        # Detect sharp correlation shifts over 20 trading days
        shift_val = s.diff(20)
        significant = shift_val[shift_val.abs() >= threshold]

        for idx, val in significant.items():
            date_str = corr_df.loc[idx, "date"]
            shifts.append({
                "date": date_str,
                "macro_key": macro_key,
                "window": window,
                "change": float(val),
                "current_correlation": float(corr_df.loc[idx, col])
            })
        return shifts
