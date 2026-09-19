import pandas as pd
import numpy as np
from typing import Dict, Any

class ScoringEngine:
    """Calculates transparent sub-scores (0-100) and composite confidence/research scores."""

    @staticmethod
    def calculate_scores(
        metal_analysis: Dict[str, Any],
        macro_regime: Dict[str, Any],
        ratio_analysis: Dict[str, Any],
        analogue_stats: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Sub-scores:
        - Technical Score (0-100)
        - Macro Score (0-100)
        - Momentum Score (0-100)
        - Volatility Score (0-100)
        - Relative Value Score (0-100)
        - Historical Pattern Score (0-100)
        """
        rsi = metal_analysis.get("rsi_14", 50.0)
        macd = metal_analysis.get("macd", 0.0)
        dist_sma200 = metal_analysis.get("dist_sma_200_pct", 0.0)
        vol_20d = metal_analysis.get("volatility_20d", 15.0)

        # Technical Score
        tech_score = 50.0
        if rsi > 50: tech_score += 15.0
        if macd > 0: tech_score += 15.0
        if dist_sma200 > 0: tech_score += 20.0
        tech_score = max(0.0, min(100.0, tech_score))

        # Momentum Score
        mom_score = 50.0
        if rsi > 60: mom_score += 25.0
        elif rsi < 40: mom_score -= 25.0
        mom_score = max(0.0, min(100.0, mom_score))

        # Macro Score
        primary_regime = macro_regime.get("primary_regime", "Neutral")
        macro_score = 50.0
        if primary_regime in ["Risk-off", "Inflationary", "Stagflationary", "Easing"]:
            macro_score = 75.0
        elif primary_regime in ["Tightening", "Risk-on"]:
            macro_score = 35.0

        # Volatility Score
        vol_score = 50.0
        if vol_20d > 25.0: vol_score = 30.0 # High risk
        elif vol_20d < 12.0: vol_score = 70.0 # Low risk

        # Relative Value Score
        rel_val_score = 50.0
        if ratio_analysis.get("is_extreme_high"):
            rel_val_score = 80.0 if metal_analysis.get("symbol") == "SILVER_USD" else 30.0
        elif ratio_analysis.get("is_extreme_low"):
            rel_val_score = 30.0 if metal_analysis.get("symbol") == "SILVER_USD" else 80.0

        # Historical Pattern Score
        pattern_score = 50.0
        fwd_10d = analogue_stats.get("forward_stats", {}).get("10d", {})
        if fwd_10d:
            pattern_score = fwd_10d.get("win_rate_pct", 50.0)

        # Composite Score
        weights = config.get("score_weights", {
            "technical": 0.25,
            "macro": 0.25,
            "momentum": 0.15,
            "volatility": 0.10,
            "relative_value": 0.15,
            "historical_pattern": 0.10
        })

        composite = (
            tech_score * weights.get("technical", 0.25) +
            macro_score * weights.get("macro", 0.25) +
            mom_score * weights.get("momentum", 0.15) +
            vol_score * weights.get("volatility", 0.10) +
            rel_val_score * weights.get("relative_value", 0.15) +
            pattern_score * weights.get("historical_pattern", 0.10)
        )

        return {
            "technical_score": round(tech_score, 1),
            "macro_score": round(macro_score, 1),
            "momentum_score": round(mom_score, 1),
            "volatility_score": round(vol_score, 1),
            "relative_value_score": round(rel_val_score, 1),
            "historical_pattern_score": round(pattern_score, 1),
            "composite_score": round(composite, 1)
        }
