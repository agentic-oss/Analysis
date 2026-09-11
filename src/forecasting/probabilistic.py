"""
Probabilistic Signals Engine.
Formulates probabilistic research signals for 1D, 3D, 5D, 10D, 20D, 60D horizons
disclaiming guaranteed outcomes.
"""

from typing import Dict, Any, List
import numpy as np


class ProbabilisticSignalEngine:
    """Combines historical statistics, technical signals, and model predictions into probabilistic research signals."""

    def generate_probabilistic_signals(
        self,
        analogue_stats: Dict[str, Any],
        technical_score: float,
        macro_score: float,
        model_forecasts: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generates research signals per horizon (1D..60D).
        Outputs direction probability, expected return, confidence, and research thesis.
        """
        horizons_signals = {}

        for horizon, stats in analogue_stats.items():
            sample_cnt = stats.get("sample_count", 0)
            hist_pos_prob = stats.get("prob_positive", 0.5)
            exp_ret = stats.get("expected_return", 0.0)

            # Combined research directional bias probability
            score_bias = (technical_score + macro_score - 100.0) / 200.0  # range -0.5 to +0.5
            combined_prob_pos = float(np.clip(hist_pos_prob * 0.6 + (0.5 + score_bias) * 0.4, 0.05, 0.95))

            if combined_prob_pos > 0.55:
                direction = "Bullish Research Signal"
            elif combined_prob_pos < 0.45:
                direction = "Bearish Research Signal"
            else:
                direction = "Neutral / Sideways Signal"

            # Confidence score calculation (0 to 100)
            sample_confidence = min(sample_cnt / 15.0, 1.0) * 40.0
            agreement_confidence = abs(combined_prob_pos - 0.5) * 120.0
            confidence_score = round(min(sample_confidence + agreement_confidence, 100.0), 1)

            reasons = [
                f"Historical analogue positive return frequency: {hist_pos_prob*100:.1f}%",
                f"Technical Score ({technical_score:.0f}/100) and Macro Score ({macro_score:.0f}/100) alignment",
                f"Sample size: {sample_cnt} historical matches",
            ]

            horizons_signals[horizon] = {
                "horizon": horizon,
                "direction": direction,
                "prob_positive": round(combined_prob_pos, 4),
                "prob_negative": round(1.0 - combined_prob_pos, 4),
                "expected_return_pct": round(exp_ret * 100.0, 2),
                "confidence_score": confidence_score,
                "disclaimer": "Conditional research statistic based on historical patterns, not a guaranteed return prediction.",
                "reasons": reasons,
            }

        return horizons_signals
