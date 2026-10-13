"""Probabilistic Forward Return Signal & Research Confidence Framework."""

from typing import Dict, Any, List


class ForwardProbabilityEngine:
    """
    Generates probabilistic forward-looking signals and research confidence scores.
    Expresses all forecasts strictly as probabilistic research signals rather than guaranteed future prices.
    """

    @staticmethod
    def generate_probabilistic_signal(
        analogue_stats: Dict[str, Any],
        technical_summary: Dict[str, Any],
        macro_summary: Dict[str, Any],
        horizons: List[int] = [1, 3, 5, 10, 20, 60]
    ) -> Dict[str, Any]:
        """Synthesizes forward probabilities, directional research bias, and explainable confidence score."""

        horizon_signals = {}
        overall_positive_probs = []

        for h in horizons:
            key = f"{h}d"
            h_stat = analogue_stats.get(key, {})
            pos_prob = h_stat.get("positive_prob_pct", 50.0)
            overall_positive_probs.append(pos_prob)

            median_ret = h_stat.get("median_return_pct", 0.0)
            mean_ret = h_stat.get("mean_return_pct", 0.0)

            horizon_signals[key] = {
                "positive_return_probability_pct": pos_prob,
                "negative_return_probability_pct": round(100.0 - pos_prob, 1),
                "expected_median_return_pct": median_ret,
                "expected_mean_return_pct": mean_ret,
                "sample_count": h_stat.get("sample_count", 0),
                "disclaimer": "Probabilistic research signal based on historical conditional statistics. Not a guaranteed price target."
            }

        avg_pos_prob = sum(overall_positive_probs) / len(overall_positive_probs) if overall_positive_probs else 50.0

        if avg_pos_prob >= 60.0:
            bias = "BULLISH"
        elif avg_pos_prob <= 40.0:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"

        # Confidence Scoring Framework (0-100)
        confidence_reasons = []
        confidence_score = 50.0

        # Reason 1: Historical sample size
        samples_10d = analogue_stats.get("10d", {}).get("sample_count", 0)
        if samples_10d >= 10:
            confidence_score += 15
            confidence_reasons.append(f"Sufficient historical analogue sample size ({samples_10d} matches)")
        else:
            confidence_reasons.append(f"Limited historical analogue sample size ({samples_10d} matches)")

        # Reason 2: Historical consistency
        if avg_pos_prob >= 65.0 or avg_pos_prob <= 35.0:
            confidence_score += 15
            confidence_reasons.append(f"Strong directional consistency across historical analogues ({avg_pos_prob:.1f}% positive frequency)")
        else:
            confidence_reasons.append(f"Mixed directional outcomes across historical analogues ({avg_pos_prob:.1f}% positive frequency)")

        # Reason 3: Technical trend alignment
        rsi = technical_summary.get("rsi_14", 50.0)
        dist_200 = technical_summary.get("dist_sma_200_pct", 0.0)
        if (bias == "BULLISH" and dist_200 > 0 and rsi < 70) or (bias == "BEARISH" and dist_200 < 0 and rsi > 30):
            confidence_score += 10
            confidence_reasons.append("Technical trend indicators aligned with research bias")
        else:
            confidence_reasons.append("Technical indicators neutrally or inversely aligned")

        # Reason 4: Macro regime alignment
        macro_regime = macro_summary.get("primary_regime", "Neutral")
        if (bias == "BULLISH" and macro_regime in ["Risk-off", "Inflationary", "Easing", "Stagflationary"]) or \
           (bias == "BEARISH" and macro_regime in ["Risk-on", "Tightening"]):
            confidence_score += 10
            confidence_reasons.append(f"Macro regime ({macro_regime}) aligns with research bias")
        else:
            confidence_reasons.append(f"Macro regime ({macro_regime}) offers neutral/mixed signal")

        confidence_score = max(0.0, min(100.0, round(confidence_score, 1)))

        return {
            "research_bias": bias,
            "confidence_score": confidence_score,
            "confidence_reasons": confidence_reasons,
            "horizons": horizon_signals
        }
