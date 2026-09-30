import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ProbabilisticForecastEngine:
    """Generates probabilistic forward-looking signals across 1D, 3D, 5D, 10D, 20D, 60D horizons."""

    def __init__(self, horizons: List[int] = [1, 3, 5, 10, 20, 60]):
        self.horizons = horizons

    def generate_forecast_signals(
        self, analogue_stats: Dict[str, Any], model_predictions: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Combines analogue return distributions and statistical model output into probabilistic research signals.
        All outputs explicitly framed as probabilistic research signals rather than guaranteed future prices.
        """
        signals = {}

        for h in self.horizons:
            h_key = f"{h}d"
            a_stat = analogue_stats.get(h_key, {})
            m_pred = model_predictions.get(h, {})

            pos_prob_analogue = a_stat.get("win_probability", 50.0) / 100.0
            pos_prob_model = m_pred.get("direction_prob", 0.50)

            # Blended direction probability (60% analogue, 40% model)
            blended_prob = 0.6 * pos_prob_analogue + 0.4 * pos_prob_model

            exp_ret = a_stat.get("mean_return_pct", 0.0)
            med_ret = a_stat.get("median_return_pct", 0.0)

            signals[h_key] = {
                "horizon_days": h,
                "probability_positive": round(blended_prob, 4),
                "probability_negative": round(1.0 - blended_prob, 4),
                "expected_return_pct": round(exp_ret, 2),
                "median_return_pct": round(med_ret, 2),
                "expected_volatility_pct": round(a_stat.get("std_pct", 0.0), 2),
                "historical_max_gain_pct": round(a_stat.get("max_gain_pct", 0.0), 2),
                "historical_max_loss_pct": round(a_stat.get("max_loss_pct", 0.0), 2),
                "disclaimer": "Probabilistic research signal based on historical conditional statistics. Not a guaranteed price prediction.",
            }

        return signals
