from typing import Dict, Any, List


class PredictionConfidenceFramework:
    """
    Evaluates research bias and prediction confidence score (0-100) with explainable reasons.
    Strictly framed as historical probabilistic research signals, not guaranteed outcomes.
    """

    @staticmethod
    def evaluate_confidence(
        composite_summary: Dict[str, Any],
        analogue_stats: Dict[str, Any],
        market_regime: Dict[str, Any],
    ) -> Dict[str, Any]:
        comp_score = composite_summary.get("composite_score", 50.0)
        reasons = []

        # Research Bias Direction
        if comp_score >= 60.0:
            research_bias = "Bullish"
            reasons.append(f"Composite score of {comp_score:.1f}/100 indicates overall bullish alignment.")
        elif comp_score <= 40.0:
            research_bias = "Bearish"
            reasons.append(f"Composite score of {comp_score:.1f}/100 indicates overall bearish alignment.")
        else:
            research_bias = "Neutral"
            reasons.append(f"Composite score of {comp_score:.1f}/100 indicates neutral/rangebound conditions.")

        # Confidence Score calculation
        base_confidence = 50.0

        # Component alignment check
        comps = composite_summary.get("components", {})
        aligned = 0
        total_comps = len(comps)
        for name, val in comps.items():
            if research_bias == "Bullish" and val >= 55.0:
                aligned += 1
            elif research_bias == "Bearish" and val <= 45.0:
                aligned += 1
            elif research_bias == "Neutral" and 40.0 <= val <= 60.0:
                aligned += 1

        alignment_pct = (aligned / total_comps) * 100.0 if total_comps > 0 else 50.0
        reasons.append(f"Pillar alignment: {aligned}/{total_comps} components agree with the research bias.")

        # Analogue consistency check
        analogue_5d = analogue_stats.get("forward_statistics", {}).get("5d", {})
        sample_count = analogue_5d.get("sample_count", 0)
        pos_prob = analogue_5d.get("pos_prob_pct", 50.0)

        if sample_count >= 5:
            reasons.append(f"Historical analogue sample size ({sample_count} observations) is robust.")
            base_confidence += 10.0
            if research_bias == "Bullish" and pos_prob >= 60.0:
                base_confidence += 15.0
                reasons.append(f"Historical analogues show {pos_prob}% positive forward returns over 5 days.")
            elif research_bias == "Bearish" and pos_prob <= 40.0:
                base_confidence += 15.0
                reasons.append(f"Historical analogues show {100-pos_prob}% negative forward returns over 5 days.")
        else:
            reasons.append("Historical analogue sample size is small (< 5), lowering confidence.")
            base_confidence -= 10.0

        # Market regime factor
        vol_regime = market_regime.get("volatility_regime", "Normal Volatility")
        if vol_regime == "High Volatility":
            base_confidence -= 10.0
            reasons.append("High volatility regime increases forecasting noise.")

        confidence_score = min(100, max(10, int(base_confidence + (alignment_pct - 50.0) * 0.3)))

        return {
            "research_bias": research_bias,
            "confidence_score": confidence_score,
            "disclaimer": "Probabilistic research signal based on historical conditions. Not a guaranteed future price prediction.",
            "reasons": reasons,
        }
