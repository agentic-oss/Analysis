import numpy as np
import pandas as pd
from typing import Dict, Any, List

class ProbabilisticForecaster:
    """Generates probabilistic forward-return statistics and confidence reasons across horizons."""

    @staticmethod
    def generate_forecasts(
        analogue_output: Dict[str, Any],
        scores: Dict[str, Any],
        horizons: List[int] = [1, 3, 5, 10, 20, 60]
    ) -> Dict[str, Any]:
        """
        Calculates probability of positive/negative return, expected return, median return,
        expected volatility, and confidence score with explicit reasons.
        """
        fwd_stats = analogue_output.get("forward_stats", {})
        sample_count = analogue_output.get("sample_count", 0)

        forecasts = {}
        for h in horizons:
            h_key = f"{h}d"
            stat = fwd_stats.get(h_key, {})

            if stat:
                prob_pos = stat.get("win_rate_pct", 50.0)
                prob_neg = stat.get("loss_rate_pct", 50.0)
                mean_ret = stat.get("mean_return_pct", 0.0)
                median_ret = stat.get("median_return_pct", 0.0)
                exp_vol = stat.get("volatility_pct", 1.0)
            else:
                prob_pos = 50.0
                prob_neg = 50.0
                mean_ret = 0.0
                median_ret = 0.0
                exp_vol = 1.0

            forecasts[h_key] = {
                "horizon_days": h,
                "probability_positive_pct": prob_pos,
                "probability_negative_pct": prob_neg,
                "expected_mean_return_pct": mean_ret,
                "median_return_pct": median_ret,
                "expected_volatility_pct": exp_vol,
                "disclaimer": "Conditional historical research signal. Not guaranteed future outcome."
            }

        # Confidence Score calculation (0-100)
        conf_score = min(100, int(sample_count * 3 + (scores.get("composite_score", 50.0) * 0.5)))
        reasons = [
            f"{sample_count} historical analogues identified in look-ahead-free search.",
            f"Composite technical/macro score at {scores.get('composite_score', 50.0)}/100.",
            f"10-day forward positive probability: {forecasts.get('10d', {}).get('probability_positive_pct', 50.0)}%"
        ]

        return {
            "forecasts": forecasts,
            "confidence_score": conf_score,
            "confidence_reasons": reasons
        }
