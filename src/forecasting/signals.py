"""
Probabilistic forward-looking research signal generator.
Frame outputs as probabilistic research signals rather than guaranteed future prices.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


class ProbabilisticSignalEngine:
    """
    Generates multi-horizon probabilistic research signals and confidence scores.
    """

    @staticmethod
    def generate_signal(
        instrument: str,
        current_price: float,
        composite_score: float,
        analogue_stats: Dict[str, Any],
        model_predictions: Dict[str, Any],
        market_regime: str,
    ) -> Dict[str, Any]:
        """
        Creates probabilistic signals for 1D, 3D, 5D, 10D, 20D, 60D horizons with explicit confidence reasoning.
        """
        horizons = [1, 3, 5, 10, 20, 60]
        signals_by_horizon = {}

        for h in horizons:
            h_key = f"{h}d"
            a_stat = analogue_stats.get(h_key, {})
            pos_prob_analogue = a_stat.get("positive_probability_pct", 50.0)
            mean_ret_analogue = a_stat.get("mean_return_pct", 0.0)

            model_prob = model_predictions.get("ensemble_positive_probability", 0.5) * 100.0

            # Combined research signal probability
            combined_prob = (0.4 * pos_prob_analogue) + (0.4 * model_prob) + (0.2 * composite_score)
            combined_prob = float(np.clip(combined_prob, 0.0, 100.0))

            direction = "Bullish" if combined_prob > 55.0 else ("Bearish" if combined_prob < 45.0 else "Neutral")

            # Confidence calculation (0-100)
            confidence_reasons = []
            sample_count = a_stat.get("sample_count", 0)

            # 1. Sample size component
            if sample_count >= 10:
                sample_conf = 25
                confidence_reasons.append(f"Sufficient historical analogue sample size ({sample_count})")
            else:
                sample_conf = 10
                confidence_reasons.append(f"Limited historical analogue sample size ({sample_count})")

            # 2. Model alignment
            model_agreement = model_predictions.get("model_agreement_ratio", 0.5)
            model_conf = int(model_agreement * 25)
            confidence_reasons.append(f"Model agreement ratio: {int(model_agreement*100)}%")

            # 3. Indicator alignment
            score_dev = abs(composite_score - 50.0)
            indicator_conf = int(np.clip(score_dev * 0.8, 0, 25))
            confidence_reasons.append(f"Multi-factor composite score: {composite_score}/100")

            # 4. Regime stability
            regime_conf = 20 if "High Volatility" not in market_regime else 10
            confidence_reasons.append(f"Market Regime: {market_regime}")

            total_confidence = sample_conf + model_conf + indicator_conf + regime_conf

            signals_by_horizon[h_key] = {
                "horizon_days": h,
                "research_bias": direction,
                "probability_positive_return_pct": round(combined_prob, 1),
                "probability_negative_return_pct": round(100.0 - combined_prob, 1),
                "historical_mean_return_pct": round(mean_ret_analogue, 2),
                "confidence_score": total_confidence,
                "confidence_reasons": confidence_reasons,
                "disclaimer": "Probabilistic research signal derived from historical conditional statistics. Not a guaranteed future price prediction.",
            }

        return {
            "instrument": instrument,
            "current_price": current_price,
            "overall_research_bias": signals_by_horizon["5d"]["research_bias"],
            "overall_confidence": signals_by_horizon["5d"]["confidence_score"],
            "horizon_signals": signals_by_horizon,
        }
