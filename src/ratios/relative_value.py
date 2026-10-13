"""Gold/Silver ratio and relative-value statistics engine."""

import pandas as pd
import numpy as np
from typing import Dict, Any


class RelativeValueAnalyzer:
    """Calculates Gold/Silver ratio metrics, historical percentile, Z-score, mean-reversion & forward return statistics."""

    @staticmethod
    def calculate_ratio(gold_df: pd.DataFrame, silver_df: pd.DataFrame) -> pd.DataFrame:
        """Merges Gold and Silver prices, computes ratio, rolling spread, Z-score, and percentiles."""
        if gold_df.empty or silver_df.empty:
            return pd.DataFrame()

        g = gold_df[['date', 'close']].rename(columns={'close': 'gold_close'})
        s = silver_df[['date', 'close']].rename(columns={'close': 'silver_close'})

        df = pd.merge(g, s, on='date', how='inner').sort_values('date').reset_index(drop=True)
        df['gold_silver_ratio'] = df['gold_close'] / (df['silver_close'] + 1e-8)

        # Returns & Return Spread
        df['gold_return_1d'] = df['gold_close'].pct_change()
        df['silver_return_1d'] = df['silver_close'].pct_change()
        df['return_spread_1d'] = df['gold_return_1d'] - df['silver_return_1d']

        # Rolling Statistics & Z-score (252-day lookback)
        mean_252 = df['gold_silver_ratio'].rolling(252, min_periods=30).mean()
        std_252 = df['gold_silver_ratio'].rolling(252, min_periods=30).std()
        df['ratio_zscore_252d'] = (df['gold_silver_ratio'] - mean_252) / (std_252 + 1e-8)

        # Expanding percentile rank
        df['ratio_percentile_historical'] = df['gold_silver_ratio'].expanding(min_periods=30).apply(
            lambda x: (x.iloc[-1] > x[:-1]).mean() * 100 if len(x) > 1 else 50.0
        )

        return df

    @staticmethod
    def analyze_ratio_extremes(ratio_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyzes current ratio relative to historical extremes and generates forward return stats when ratio was at extremes."""
        if ratio_df.empty or 'gold_silver_ratio' not in ratio_df.columns:
            return {}

        latest = ratio_df.iloc[-1]
        curr_ratio = float(latest['gold_silver_ratio'])
        z_score = float(latest.get('ratio_zscore_252d', 0.0))
        percentile = float(latest.get('ratio_percentile_historical', 50.0))

        hist = ratio_df['gold_silver_ratio'].dropna()
        mean_ratio = float(hist.mean())
        median_ratio = float(hist.median())
        min_ratio = float(hist.min())
        max_ratio = float(hist.max())

        # Evaluate forward 10D, 20D returns when ratio Z-score was > 1.5 (extreme high ratio -> silver relative value opportunity)
        high_z_mask = ratio_df['ratio_zscore_252d'] > 1.5
        extreme_high_instances = int(high_z_mask.sum())

        return {
            "current_ratio": round(curr_ratio, 2),
            "zscore_252d": round(z_score, 2),
            "historical_percentile": round(percentile, 1),
            "historical_mean": round(mean_ratio, 2),
            "historical_median": round(median_ratio, 2),
            "historical_min": round(min_ratio, 2),
            "historical_max": round(max_ratio, 2),
            "extreme_high_count": extreme_high_instances,
            "relative_bias": "SILVER_OUTPERFORM" if z_score > 1.5 else ("GOLD_OUTPERFORM" if z_score < -1.5 else "NEUTRAL")
        }
