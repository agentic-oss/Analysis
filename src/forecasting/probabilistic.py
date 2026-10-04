import pandas as pd
import numpy as np
from typing import Dict, Any, List

class ForwardProbabilityEngine:
    """Generates probabilistic estimates of future price directions and returns based on historical conditional statistics."""

    @staticmethod
    def generate_probabilistic_signals(
        analogue_stats: Dict[str, Any],
        horizons: List[int] = [1, 3, 5, 10, 20, 60]
    ) -> Dict[str, Any]:
        """Convert analogue forward return distributions into research signals for each horizon."""
        signals = {}

        for h in horizons:
            h_key = f"{h}d"
            stats = analogue_stats.get(h_key)

            if not stats:
                signals[h_key] = {
                    "horizon_days": h,
                    "prob_positive": 0.5,
                    "prob_negative": 0.5,
                    "expected_return_pct": 0.0,
                    "median_return_pct": 0.0,
                    "volatility_pct": 0.0,
                    "historical_sample_count": 0,
                    "signal_direction": "Neutral",
                    "disclaimer": "Probabilistic research signal based on historical analogues, not a guaranteed forecast."
                }
                continue

            prob_pos = stats.get("prob_positive", 0.5)
            mean_ret = stats.get("mean_return", 0.0)

            if prob_pos >= 0.60 and mean_ret > 0:
                direction = "Bullish"
            elif prob_pos <= 0.40 and mean_ret < 0:
                direction = "Bearish"
            else:
                direction = "Neutral"

            signals[h_key] = {
                "horizon_days": h,
                "prob_positive": round(prob_pos, 4),
                "prob_negative": round(1.0 - prob_pos, 4),
                "expected_return_pct": round(mean_ret * 100.0, 2),
                "median_return_pct": round(stats.get("median_return", 0.0) * 100.0, 2),
                "volatility_pct": round(stats.get("volatility", 0.0) * 100.0, 2),
                "historical_sample_count": stats.get("sample_count", 0),
                "signal_direction": direction,
                "disclaimer": "Probabilistic research signal based on historical analogues, not a guaranteed forecast."
            }

        return signals
