import numpy as np
import pandas as pd
from typing import Dict, Any, List

class PredictionConfidenceFramework:
    """
    Evaluates probabilistic signal confidence based on historical sample size,
    model agreement, signal consistency, feature stability, and volatility regime.
    Outputs transparent score 0-100 with explainable reasons.
    """

    def calculate_confidence(
        self,
        analogue_sample_size: int,
        analogue_positive_prob: float,
        model_agreement_pct: float,
        macro_alignment: bool,
        current_volatility: float
    ) -> Dict[str, Any]:

        reasons = []
        score = 50.0 # baseline

        # Sample size factor
        if analogue_sample_size >= 20:
            score += 15.0
            reasons.append(f"Robust historical sample size ({analogue_sample_size} analogues)")
        elif analogue_sample_size >= 8:
            score += 8.0
            reasons.append(f"Moderate historical sample size ({analogue_sample_size} analogues)")
        else:
            score -= 10.0
            reasons.append(f"Small historical sample size ({analogue_sample_size} analogues)")

        # Historical consistency
        conviction = abs(analogue_positive_prob - 50.0)
        if conviction >= 20.0:
            score += 15.0
            reasons.append(f"High historical signal directional conviction ({analogue_positive_prob:.1f}% positive frequency)")
        elif conviction >= 10.0:
            score += 8.0
            reasons.append(f"Moderate historical directional conviction ({analogue_positive_prob:.1f}% positive frequency)")

        # Model agreement
        if model_agreement_pct >= 0.75:
            score += 15.0
            reasons.append(f"Strong model alignment ({model_agreement_pct*100:.0f}% model agreement)")
        elif model_agreement_pct <= 0.25:
            score -= 10.0
            reasons.append("Conflicting model predictions")

        # Macro alignment
        if macro_alignment:
            score += 10.0
            reasons.append("Macro regime aligns with technical research signal")

        # Volatility penalty
        if current_volatility > 0.25: # high vol penalty
            score -= 10.0
            reasons.append("High volatility environment increases outcome uncertainty")

        final_score = max(0.0, min(100.0, round(score, 1)))

        return {
            "confidence_score": final_score,
            "reasons": reasons,
            "disclaimer": "All price predictions and forecasting outputs are probabilistic research signals, not guaranteed future prices."
        }
