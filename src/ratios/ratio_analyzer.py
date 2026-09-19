import pandas as pd
import numpy as np
from typing import Dict, Any

class RatioAnalyzer:
    """Analyzes Gold/Silver ratio, relative value, and return spreads."""

    @staticmethod
    def calculate_gold_silver_ratio(
        gold_df: pd.DataFrame,
        silver_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculates Gold/Silver ratio, historical percentile, Z-score, mean reversion,
        and relative performance stats.
        """
        if gold_df.empty or silver_df.empty:
            return {}

        merged = pd.merge(
            gold_df[['date', 'close', 'return_1d', 'return_5d', 'return_20d']],
            silver_df[['date', 'close', 'return_1d', 'return_5d', 'return_20d']],
            on='date',
            suffixes=('_gold', '_silver')
        ).sort_values('date').reset_index(drop=True)

        if merged.empty:
            return {}

        merged['ratio'] = merged['close_gold'] / (merged['close_silver'] + 1e-10)
        merged['spread_1d'] = merged['return_1d_gold'] - merged['return_1d_silver']

        latest_ratio = float(merged['ratio'].iloc[-1])
        ratio_history = merged['ratio']

        mean_ratio = float(ratio_history.mean())
        std_ratio = float(ratio_history.std()) if len(ratio_history) > 1 else 1.0
        z_score = (latest_ratio - mean_ratio) / (std_ratio + 1e-10)

        percentile = float((ratio_history <= latest_ratio).mean() * 100)

        # Extreme conditions
        extreme_high = z_score > 2.0
        extreme_low = z_score < -2.0

        return {
            "current_ratio": round(latest_ratio, 2),
            "historical_mean": round(mean_ratio, 2),
            "historical_std": round(std_ratio, 2),
            "z_score": round(float(z_score), 2),
            "percentile": round(percentile, 1),
            "is_extreme_high": extreme_high,
            "is_extreme_low": extreme_low,
            "latest_spread_1d_pct": round(float(merged['spread_1d'].iloc[-1]), 2),
            "ratio_series": merged[['date', 'ratio', 'spread_1d']]
        }
