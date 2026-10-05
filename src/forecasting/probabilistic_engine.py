"""Probabilistic forward signal and confidence score generator."""

from typing import Dict, Any, List
import numpy as np


class ProbabilisticForecaster:
    """Generates probabilistic return signals and confidence metrics."""

    def generate_forward_signals(
        self,
        analogue_stats: Dict[str, Any],
        model_predictions: Dict[str, Any],
        composite_score: float,
        current_volatility: float
    ) -> Dict[str, Any]:
        """
        Generates probabilistic forecasts across horizons (1D, 3D, 5D, 10D, 20D, 60D)
        framed strictly as probabilistic research signals rather than guaranteed future prices.
        """
        horizons = [1, 3, 5, 10, 20, 60]
        signals = {}

        for h in horizons:
            h_key = f"{h}d"
            fwd_info = analogue_stats.get('forward_stats', {}).get(h_key, {})
            sample_cnt = fwd_info.get('sample_count', 0)
            pos_prob = fwd_info.get('positive_probability', 0.5)
            mean_ret = fwd_info.get('mean_return', 0.0)
            median_ret = fwd_info.get('median_return', 0.0)

            # Model consensus direction
            m_pred = model_predictions.get(h_key, {})
            m_dir = m_pred.get('direction_prob', 0.5)

            # Blended positive probability
            blended_prob = 0.6 * pos_prob + 0.4 * m_dir

            signals[h_key] = {
                "horizon_days": h,
                "probability_positive": round(float(blended_prob), 4),
                "probability_negative": round(float(1.0 - blended_prob), 4),
                "expected_return": round(float(mean_ret), 6),
                "median_return": round(float(median_ret), 6),
                "historical_sample_count": sample_cnt,
                "research_disclaimer": "Probabilistic historical conditional statistic only; not a price guarantee."
            }

        # Calculate confidence score (0-100) and explainable reasons
        confidence_score, reasons = self._calculate_confidence(
            analogue_stats=analogue_stats,
            composite_score=composite_score,
            current_volatility=current_volatility,
            model_predictions=model_predictions
        )

        return {
            "forecast_signals": signals,
            "confidence": {
                "score": round(confidence_score, 2),
                "reasons": reasons
            }
        }

    def _calculate_confidence(
        self,
        analogue_stats: Dict[str, Any],
        composite_score: float,
        current_volatility: float,
        model_predictions: Dict[str, Any]
    ) -> tuple[float, List[str]]:
        reasons = []
        base_confidence = 50.0

        # 1. Sample size check
        fwd_10d = analogue_stats.get('forward_stats', {}).get('10d', {})
        n_samples = fwd_10d.get('sample_count', 0)
        if n_samples >= 10:
            base_confidence += 15.0
            reasons.append(f"Robust historical sample size ({n_samples} analogues found)")
        else:
            reasons.append(f"Small historical sample size ({n_samples} analogues found)")

        # 2. Score alignment
        if composite_score > 65 or composite_score < 35:
            base_confidence += 15.0
            reasons.append("Multi-factor composite scores show clear directional alignment")
        else:
            reasons.append("Multi-factor composite scores are neutral")

        # 3. Volatility stability
        if current_volatility < 0.20:
            base_confidence += 10.0
            reasons.append("Low/normal volatility regime supports signal stability")
        else:
            base_confidence -= 10.0
            reasons.append("High volatility regime increases forecast dispersion")

        final_score = float(np.clip(base_confidence, 10.0, 95.0))
        return final_score, reasons
