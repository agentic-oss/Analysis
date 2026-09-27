import pandas as pd
import numpy as np
from typing import Dict, Any, List

class MacroAnalyzer:
    """Calculates cross-market rolling correlations and macro driver dynamics."""

    def __init__(self, windows: List[int] = [20, 60, 120, 252]):
        self.windows = windows

    def calculate_correlations(self, master_df: pd.DataFrame, target_col: str, factor_cols: List[str]) -> pd.DataFrame:
        """
        Calculates rolling correlations between a target asset (e.g., GOLD_close or SILVER_close)
        and multiple macro driver assets across various windows.
        """
        df = master_df.copy()
        target_returns = df[target_col].astype(float).pct_change()

        for factor in factor_cols:
            if factor not in df.columns:
                continue
            factor_returns = df[factor].astype(float).pct_change()

            for w in self.windows:
                corr_col = f"corr_{target_col}_vs_{factor}_{w}d"
                df[corr_col] = target_returns.rolling(window=w, min_periods=10).corr(factor_returns)

        return df
