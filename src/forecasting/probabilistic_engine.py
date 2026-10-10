import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class ProbabilisticEngine:
    """
    Generates multi-horizon probabilistic research signals.
    Combines historical analogue conditional statistics with ML model consensus.
    Expresses predictions strictly as probabilistic research signals rather than guaranteed future prices.
    """

    @staticmethod
    def generate_probabilistic_signals(horizon_days: int, analogue_stats: Optional[Dict[str, Any]], model_pred: Dict[str, Any], hist_returns: pd.Series) -> Dict[str, Any]:
        """
        Calculates direction probabilities, expected return, median return, expected volatility, and confidence score.
        """
        analogue_h = analogue_stats.get(f"{horizon_days}d") if analogue_stats else None

        analogue_prob = analogue_h["prob_positive_pct"] / 100.0 if analogue_h and "prob_positive_pct" in analogue_h else 0.50
        analogue_exp_ret = analogue_h["mean_return_pct"] if analogue_h and "mean_return_pct" in analogue_h else 0.0

        model_prob = model_pred.get("direction_prob", 0.50)
        model_exp_ret = model_pred.get("predicted_return_pct", 0.0)

        # Weighted Probabilistic Combination
        prob_positive = round((0.60 * analogue_prob) + (0.40 * model_prob), 4)
        prob_negative = round(1.0 - prob_positive, 4)

        expected_return_pct = round((0.60 * analogue_exp_ret) + (0.40 * model_exp_ret), 2)

        # Volatility & Historical Range
        if len(hist_returns) >= horizon_days:
            horizon_vol = float(hist_returns.rolling(horizon_days).std().iloc[-1] * np.sqrt(252 / horizon_days) * 100)
            if np.isnan(horizon_vol):
                horizon_vol = 3.5
        else:
            horizon_vol = 3.5

        # Confidence Calculation (0 - 100 scale based on sample size, alignment, and volatility)
        sample_size = analogue_h["sample_count"] if analogue_h and "sample_count" in analogue_h else 0
        alignment_bonus = 20.0 if (prob_positive > 0.60 or prob_positive < 0.40) else 0.0
        sample_bonus = min(30.0, sample_size * 5.0)

        confidence_score = min(100.0, round(40.0 + sample_bonus + alignment_bonus, 1))

        research_bias = "Bullish Signal" if prob_positive >= 0.55 else ("Bearish Signal" if prob_positive <= 0.45 else "Neutral Signal")

        return {
            "horizon_days": horizon_days,
            "research_bias": research_bias,
            "prob_positive_return": round(prob_positive * 100, 1),
            "prob_negative_return": round(prob_negative * 100, 1),
            "expected_return_pct": expected_return_pct,
            "expected_volatility_pct": round(horizon_vol, 2),
            "confidence_score": confidence_score,
            "disclaimer": "Probabilistic research signal based on historical conditional statistics and model outputs. Not a guaranteed future price prediction."
        }
