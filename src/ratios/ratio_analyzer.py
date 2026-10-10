import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class RatioAnalyzer:
    """
    Gold/Silver ratio and relative-value analysis module.
    Calculates current ratio, historical percentile, z-score, mean reversion signals,
    and rolling spread volatility.
    """

    @staticmethod
    def calculate_ratio_metrics(gold_df: pd.DataFrame, silver_df: pd.DataFrame) -> Dict[str, Any]:
        if gold_df.empty or silver_df.empty or "Close" not in gold_df.columns or "Close" not in silver_df.columns:
            return {}

        combined = pd.concat([gold_df["Close"].rename("gold"), silver_df["Close"].rename("silver")], axis=1).dropna()
        if combined.empty:
            return {}

        ratio_series = combined["gold"] / combined["silver"]
        current_ratio = float(ratio_series.iloc[-1])

        # Historical metrics (2520 trading days ~ 10 years or max available)
        lookback = min(len(ratio_series), 2520)
        hist_series = ratio_series.iloc[-lookback:]

        mean_ratio = float(hist_series.mean())
        std_ratio = float(hist_series.std())
        z_score = float((current_ratio - mean_ratio) / std_ratio) if std_ratio > 0 else 0.0

        percentile = float((hist_series < current_ratio).mean() * 100)

        # Spread & returns differential
        gold_ret = combined["gold"].pct_change()
        silver_ret = combined["silver"].pct_change()
        ret_diff_1d = float((gold_ret - silver_ret).iloc[-1] * 100)
        ret_diff_20d = float(((combined["gold"].pct_change(20) - combined["silver"].pct_change(20)).iloc[-1]) * 100)

        # Mean reversion condition detection
        mean_reversion_signal = "Neutral"
        if z_score > 2.0 or percentile > 90.0:
            mean_reversion_signal = "Extreme High GSR - Favor Silver Outperformance"
        elif z_score < -2.0 or percentile < 10.0:
            mean_reversion_signal = "Extreme Low GSR - Favor Gold Outperformance"

        return {
            "current_gold_silver_ratio": current_ratio,
            "historical_mean": mean_ratio,
            "historical_std": std_ratio,
            "z_score": z_score,
            "percentile": percentile,
            "ret_diff_1d_pct": ret_diff_1d,
            "ret_diff_20d_pct": ret_diff_20d,
            "mean_reversion_signal": mean_reversion_signal,
            "series": ratio_series
        }
