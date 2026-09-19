import pandas as pd
import numpy as np
from typing import Dict, Any

class MacroAnalyzer:
    """Analyzes macroeconomic drivers and cross-market rolling correlations."""

    @staticmethod
    def calculate_cross_correlations(
        target_series: pd.Series,
        macro_df: pd.DataFrame,
        windows: list = [20, 60, 120, 252]
    ) -> Dict[str, Dict[str, float]]:
        """
        Computes rolling correlations between target series (e.g. Gold close)
        and macro driver series over specified lookback windows.
        """
        correlations = {}
        if target_series.empty or macro_df.empty:
            return correlations

        for col in macro_df.columns:
            if col in ['date', 'symbol', 'ticker', 'retrieved_at', 'quality_status']:
                continue
            correlations[col] = {}
            for w in windows:
                if len(target_series) >= w:
                    s1 = target_series.tail(w)
                    s2 = macro_df[col].tail(w)
                    corr = s1.corr(s2)
                    correlations[col][f"{w}d"] = round(float(corr), 4) if not pd.isna(corr) else 0.0
                else:
                    correlations[col][f"{w}d"] = 0.0
        return correlations
