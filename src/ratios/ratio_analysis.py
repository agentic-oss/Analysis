import numpy as np
import pandas as pd
from typing import Dict, Any, List

class GoldSilverRatioAnalyzer:
    """Analyzes Gold/Silver ratio, z-scores, percentiles, mean-reversion, and spread statistics."""

    def __init__(self, lookback_window: int = 252):
        self.lookback_window = lookback_window

    def calculate_ratio_metrics(
        self,
        gold_prices: pd.Series,
        silver_prices: pd.Series
    ) -> pd.DataFrame:
        """Calculates ratio, returns spread, z-score, and rolling percentiles."""
        df = pd.DataFrame({'gold': gold_prices, 'silver': silver_prices}).dropna()
        df['ratio'] = df['gold'] / df['silver']

        # Gold vs Silver return spread
        df['gold_return_1d'] = df['gold'].pct_change()
        df['silver_return_1d'] = df['silver'].pct_change()
        df['spread_1d'] = df['gold_return_1d'] - df['silver_return_1d']

        # Rolling ratio stats
        rolling_mean = df['ratio'].rolling(window=self.lookback_window).mean()
        rolling_std = df['ratio'].rolling(window=self.lookback_window).std()
        df['ratio_sma_252'] = rolling_mean
        df['ratio_zscore'] = (df['ratio'] - rolling_mean) / (rolling_std + 1e-10)

        # Percentile rank in rolling lookback
        def get_percentile(s):
            if len(s) < 20:
                return 50.0
            val = s.iloc[-1]
            return float((s < val).mean() * 100.0)

        df['ratio_percentile'] = df['ratio'].rolling(window=self.lookback_window).apply(get_percentile, raw=False)
        return df

    def analyze_spread_extreme_outcomes(
        self,
        ratio_df: pd.DataFrame,
        extreme_z_threshold: float = 2.0,
        forward_horizons: List[int] = [1, 3, 5, 10, 20, 60]
    ) -> Dict[str, Any]:
        """
        Calculates forward returns for gold, silver, and ratio after extreme ratio Z-scores.
        """
        df = ratio_df.copy()

        # Calculate forward returns
        for h in forward_horizons:
            df[f'gold_fwd_{h}d'] = df['gold'].pct_change(h).shift(-h)
            df[f'silver_fwd_{h}d'] = df['silver'].pct_change(h).shift(-h)
            df[f'ratio_fwd_{h}d'] = df['ratio'].pct_change(h).shift(-h)

        high_extremes = df[df['ratio_zscore'] >= extreme_z_threshold]
        low_extremes = df[df['ratio_zscore'] <= -extreme_z_threshold]

        results = {
            "high_extreme_z_count": len(high_extremes),
            "low_extreme_z_count": len(low_extremes),
            "high_extreme_outcomes": {},
            "low_extreme_outcomes": {}
        }

        for h in forward_horizons:
            if len(high_extremes) > 0:
                results["high_extreme_outcomes"][f'{h}d'] = {
                    "gold_mean_return": float(high_extremes[f'gold_fwd_{h}d'].mean()),
                    "silver_mean_return": float(high_extremes[f'silver_fwd_{h}d'].mean()),
                    "ratio_mean_change": float(high_extremes[f'ratio_fwd_{h}d'].mean()),
                }
            if len(low_extremes) > 0:
                results["low_extreme_outcomes"][f'{h}d'] = {
                    "gold_mean_return": float(low_extremes[f'gold_fwd_{h}d'].mean()),
                    "silver_mean_return": float(low_extremes[f'silver_fwd_{h}d'].mean()),
                    "ratio_mean_change": float(low_extremes[f'ratio_fwd_{h}d'].mean()),
                }

        return results
