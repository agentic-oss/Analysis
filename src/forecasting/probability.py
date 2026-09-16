from typing import Dict, Any, List
import numpy as np
import pandas as pd


class ForwardProbabilityEngine:
    """Generates probabilistic forward estimates explicitly labeled as historical conditional statistics."""

    @staticmethod
    def generate_probabilities(
        analogue_output: Dict[str, Any],
        horizons: List[int] = [1, 3, 5, 10, 20, 60]
    ) -> Dict[str, Any]:
        results = {}
        fwd_stats = analogue_output.get("forward_statistics", {})

        for h in horizons:
            key = f"{h}d"
            stats = fwd_stats.get(key, {})
            pos_prob = stats.get("prob_positive", 0.5)
            mean_ret = stats.get("mean_return", 0.0)
            median_ret = stats.get("median_return", 0.0)

            bias = "Bullish" if pos_prob > 0.55 else ("Bearish" if pos_prob < 0.45 else "Neutral")

            results[key] = {
                "horizon_days": h,
                "research_bias": bias,
                "probability_positive": pos_prob,
                "probability_negative": round(1.0 - pos_prob, 4),
                "expected_return": mean_ret,
                "median_return": median_ret,
                "expected_volatility": stats.get("expected_volatility", 0.0),
                "disclaimer": "Historical conditional statistics. Not a guaranteed future outcome.",
            }

        return results
