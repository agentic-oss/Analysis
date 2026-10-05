"""Gold/Silver ratio and relative value analysis."""

import numpy as np
import pandas as pd
from typing import Dict, Any


class GoldSilverRatioAnalyzer:
    """Calculates Gold/Silver ratio, Z-score, percentiles, mean reversion, and spread statistics."""

    @staticmethod
    def calculate_ratio_metrics(gold_df: pd.DataFrame, silver_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates ratio timeseries and current metrics."""
        if gold_df.empty or silver_df.empty:
            return {}

        g = gold_df[['date', 'close']].rename(columns={'close': 'gold'}).sort_values('date')
        s = silver_df[['date', 'close']].rename(columns={'close': 'silver'}).sort_values('date')

        merged = pd.merge(g, s, on='date', how='inner')
        if merged.empty:
            return {}

        merged['ratio'] = merged['gold'] / merged['silver']
        merged['gold_ret'] = merged['gold'].pct_change()
        merged['silver_ret'] = merged['silver'].pct_change()
        merged['spread_ret'] = merged['gold_ret'] - merged['silver_ret']

        ratio_s = merged['ratio']
        current_ratio = float(ratio_s.iloc[-1])
        mean_ratio = float(ratio_s.mean())
        std_ratio = float(ratio_s.std())
        z_score = float((current_ratio - ratio_s.rolling(252, min_periods=20).mean().iloc[-1]) / (ratio_s.rolling(252, min_periods=20).std().iloc[-1] + 1e-8)) if len(ratio_s) >= 20 else 0.0

        percentile = float((ratio_s < current_ratio).mean() * 100.0)

        # Forward return analysis of extreme ratio events (|z| > 1.5)
        extreme_episodes = merged[merged['ratio'].sub(merged['ratio'].rolling(252, min_periods=20).mean()).abs() > 1.5 * merged['ratio'].rolling(252, min_periods=20).std()]

        return {
            "current_ratio": round(current_ratio, 4),
            "mean_ratio_historical": round(mean_ratio, 4),
            "std_ratio_historical": round(std_ratio, 4),
            "z_score_252d": round(z_score, 4),
            "percentile_rank": round(percentile, 2),
            "extreme_episodes_count": len(extreme_episodes),
            "spread_1d": float(merged['spread_ret'].iloc[-1]) if pd.notna(merged['spread_ret'].iloc[-1]) else 0.0,
            "timeseries": merged
        }
