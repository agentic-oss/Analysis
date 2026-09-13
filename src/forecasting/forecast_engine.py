import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class ForecastEngine:
    """Generates probabilistic research signals and confidence scores across multiple horizons."""

    FORWARD_HORIZONS = [1, 3, 5, 10, 20, 60]

    def generate_probabilistic_signals(
        self,
        symbol: str,
        analogue_results: Dict[str, Any],
        technical_score: float,
        macro_score: float
    ) -> Dict[str, Any]:
        """
        Combines historical analogue frequencies, technical indicators, and macro alignment
        into probabilistic research forecasts.
        """
        fwd_stats = analogue_results.get("forward_statistics", {})
        signals = {}

        for h in self.FORWARD_HORIZONS:
            h_key = f'{h}d'
            h_stat = fwd_stats.get(h_key, {})

            pos_prob = h_stat.get("positive_probability", 0.50)
            mean_ret = h_stat.get("mean_return", 0.0)
            med_ret = h_stat.get("median_return", 0.0)
            exp_vol = h_stat.get("expected_volatility", 0.02)
            sample_count = h_stat.get("sample_count", 0)

            # Directional research bias
            if pos_prob > 0.58:
                direction = "Bullish"
            elif pos_prob < 0.42:
                direction = "Bearish"
            else:
                direction = "Neutral"

            # Confidence Score Calculation (0 - 100)
            # Based on sample size, model agreement, technical/macro alignment
            sample_weight = min(sample_count / 15.0, 1.0) * 30.0
            freq_weight = abs(pos_prob - 0.50) * 2.0 * 40.0  # 0 to 40
            alignment_weight = (30.0 if (direction == "Bullish" and technical_score > 50 and macro_score > 50) or
                                       (direction == "Bearish" and technical_score < 50 and macro_score < 50) else 15.0)

            confidence_score = round(min(sample_weight + freq_weight + alignment_weight, 100.0), 1)

            reasons = [
                f"{pos_prob*100:.1f}% positive forward-return frequency among {sample_count} historical analogues",
                f"Technical score: {technical_score}/100, Macro score: {macro_score}/100",
                f"Expected median return: {med_ret*100:+.2f}%, Expected volatility: {exp_vol*100:.2f}%"
            ]

            signals[h_key] = {
                "horizon": f"{h} trading days",
                "predicted_direction": direction,
                "positive_return_probability": round(pos_prob, 4),
                "negative_return_probability": round(1.0 - pos_prob, 4),
                "expected_mean_return": mean_ret,
                "expected_median_return": med_ret,
                "expected_volatility": exp_vol,
                "confidence_score": confidence_score,
                "reasons": reasons,
                "disclaimer": "Historical conditional statistics only; not a guaranteed outcome."
            }

        return {
            "symbol": symbol,
            "horizon_signals": signals
        }
