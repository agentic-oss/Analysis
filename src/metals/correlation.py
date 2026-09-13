import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class MetalCorrelationAnalyzer:
    """Analyzes multi-asset rolling correlations for Gold and Silver against macro drivers."""

    def __init__(self, windows: List[int] = [20, 60, 120, 252]):
        self.windows = windows

    def calculate_cross_correlations(
        self,
        metal_prices: pd.Series,
        macro_df: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculates rolling correlations between a metal (e.g., Gold) and a set of macro series
        (e.g., DXY, US10Y, Real Yields, Oil, Equities, VIX) for configured windows.
        Returns dictionary mapping window size (e.g. '20d') to DataFrame of rolling correlations.
        """
        results = {}
        metal_returns = metal_prices.pct_change()

        for w in self.windows:
            corr_df = pd.DataFrame(index=metal_prices.index)
            for col in macro_df.columns:
                macro_returns = macro_df[col].pct_change()
                corr_df[f'corr_{col}'] = metal_returns.rolling(window=w).corr(macro_returns)
            results[f'{w}d'] = corr_df

        return results

    def detect_correlation_regime_shifts(
        self,
        rolling_corrs: pd.DataFrame,
        threshold_change: float = 0.4
    ) -> List[Dict[str, Any]]:
        """
        Detects significant shift in correlation (e.g. Gold vs DXY shifting from -0.8 to +0.2).
        """
        shifts = []
        for col in rolling_corrs.columns:
            series = rolling_corrs[col].dropna()
            if len(series) < 20:
                continue
            recent_val = series.iloc[-1]
            prev_val = series.iloc[-20]  # 20 trading days prior
            diff = recent_val - prev_val
            if abs(diff) >= threshold_change:
                shifts.append({
                    "driver": col.replace('corr_', ''),
                    "current_corr": float(recent_val),
                    "previous_20d_corr": float(prev_val),
                    "change": float(diff),
                    "regime_shift_type": "breakdown" if diff < 0 else "decoupling/realignment"
                })
        return shifts
