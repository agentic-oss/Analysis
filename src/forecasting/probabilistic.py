"""
Probabilistic Forward Forecasting Engine.
Framed strictly as historical conditional probabilistic research signals.
"""
from typing import Dict, List, Any
import numpy as np


class ProbabilisticForecaster:
    """
    Generates forward research signals across multiple horizons (1D, 3D, 5D, 10D, 20D, 60D).
    All forecasts are explicitly framed as historical probabilistic statistics.
    """

    @staticmethod
    def generate_probabilistic_signal(
        symbol: str,
        analogue_forward_stats: Dict[str, Any],
        scores: Dict[str, Any],
        horizons: List[int] = [1, 3, 5, 10, 20, 60],
    ) -> Dict[str, Any]:

        comp_score = scores.get("composite_score", 50.0)

        # Overall Research Bias
        if comp_score >= 65.0:
            bias = "Bullish Research Bias"
        elif comp_score <= 35.0:
            bias = "Bearish Research Bias"
        else:
            bias = "Neutral / Mixed Research Bias"

        horizon_signals = {}
        for h in horizons:
            key = f"{h}d"
            h_stat = analogue_forward_stats.get(key, {})

            base_pos_prob = float(h_stat.get("positive_probability_pct", 50.0)) if h_stat else 50.0
            mean_ret = float(h_stat.get("mean_return_pct", 0.0)) if h_stat else 0.0
            median_ret = float(h_stat.get("median_return_pct", 0.0)) if h_stat else 0.0

            # Adjust probability slightly with composite score factor
            score_adj = (comp_score - 50.0) * 0.2
            final_pos_prob = max(10.0, min(90.0, base_pos_prob + score_adj))
            final_neg_prob = 100.0 - final_pos_prob

            horizon_signals[key] = {
                "horizon_days": h,
                "probability_positive_return_pct": round(final_pos_prob, 1),
                "probability_negative_return_pct": round(final_neg_prob, 1),
                "expected_median_return_pct": round(median_ret, 2),
                "expected_mean_return_pct": round(mean_ret, 2),
                "historical_sample_count": h_stat.get("sample_count", 0) if h_stat else 0,
            }

        return {
            "symbol": symbol,
            "research_bias": bias,
            "composite_score": comp_score,
            "disclaimer": "All price predictions and forecasting outputs are probabilistic research signals based on historical conditional statistics, NOT guaranteed future outcomes.",
            "horizons": horizon_signals,
        }
