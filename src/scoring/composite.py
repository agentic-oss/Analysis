from typing import Dict, Any

class ScoringSystem:
    """
    Transparent multi-factor scoring system (0-100 scores) for technical,
    macro, momentum, volatility, relative value, historical pattern, and composite scores.
    """

    def calculate_scores(
        self,
        rsi_14: float,
        dist_sma_20: float,
        dist_sma_200: float,
        macd_hist: float,
        macro_regime: str,
        gsr_zscore: float,
        analogue_pos_prob: float,
        hist_vol_20: float
    ) -> Dict[str, float]:

        # Technical Score (0-100)
        tech_score = 50.0
        if dist_sma_20 > 0: tech_score += 15.0
        if dist_sma_200 > 0: tech_score += 15.0
        if macd_hist > 0: tech_score += 10.0
        if rsi_14 > 50: tech_score += 10.0
        tech_score = max(0.0, min(100.0, tech_score))

        # Macro Score (0-100)
        macro_score = 50.0
        if macro_regime in ["Inflationary", "Stagflationary", "Risk-Off", "Easing"]:
            macro_score = 75.0
        elif macro_regime in ["Tightening", "Disinflationary"]:
            macro_score = 30.0

        # Momentum Score (0-100)
        mom_score = max(0.0, min(100.0, (rsi_14 / 100.0) * 100.0))

        # Volatility Score (0-100, higher means calmer/healthier trend environment)
        vol_score = max(0.0, min(100.0, 100.0 - (hist_vol_20 * 300.0)))

        # Relative Value Score (0-100)
        rv_score = max(0.0, min(100.0, 50.0 - (gsr_zscore * 20.0)))

        # Historical Pattern Score (0-100)
        pattern_score = analogue_pos_prob

        # Composite Weighted Score
        composite = (
            0.25 * tech_score +
            0.25 * macro_score +
            0.15 * mom_score +
            0.10 * vol_score +
            0.15 * rv_score +
            0.10 * pattern_score
        )

        return {
            "technical_score": round(tech_score, 1),
            "macro_score": round(macro_score, 1),
            "momentum_score": round(mom_score, 1),
            "volatility_score": round(vol_score, 1),
            "relative_value_score": round(rv_score, 1),
            "historical_pattern_score": round(pattern_score, 1),
            "composite_score": round(composite, 1)
        }
