"""Transparent composite scoring system for metals research signals."""

import pandas as pd
import numpy as np
from typing import Dict, Any


class CompositeScorer:
    """
    Calculates individual factor scores (0–100) and weighted composite score:
    Technical, Macro, Momentum, Volatility, Relative Value, Historical Pattern, Model.
    All components are preserved for explainability.
    """

    @staticmethod
    def calculate_scores(
        technical_data: Dict[str, Any],
        macro_data: Dict[str, Any],
        ratio_data: Dict[str, Any],
        analogue_data: Dict[str, Any],
        weights: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """Calculates 0-100 sub-scores and composite score."""

        if weights is None:
            weights = {
                "technical": 0.25,
                "macro": 0.20,
                "momentum": 0.15,
                "volatility": 0.10,
                "relative_value": 0.15,
                "historical_pattern": 0.15
            }

        # 1. Technical Score (0-100)
        dist_200 = technical_data.get("dist_sma_200_pct", 0.0)
        tech_score = 50.0 + min(25.0, max(-25.0, dist_200 * 2))

        # 2. Macro Score (0-100)
        macro_regime = macro_data.get("primary_regime", "Neutral")
        macro_score_map = {
            "Risk-off": 80.0,
            "Inflationary": 75.0,
            "Stagflationary": 85.0,
            "Easing": 70.0,
            "Neutral": 50.0,
            "Risk-on": 35.0,
            "Tightening": 30.0,
            "Deflationary": 25.0
        }
        macro_score = macro_score_map.get(macro_regime, 50.0)

        # 3. Momentum Score (0-100)
        rsi = technical_data.get("rsi_14", 50.0)
        macd = technical_data.get("macd", 0.0)
        mom_score = rsi

        # 4. Volatility Score (0-100)
        vol_pct = technical_data.get("volatility_20d_pct", 15.0)
        vol_score = max(0.0, min(100.0, 100.0 - (vol_pct * 2.5)))

        # 5. Relative Value Score (0-100)
        ratio_z = ratio_data.get("zscore_252d", 0.0)
        rv_score = max(0.0, min(100.0, 50.0 - (ratio_z * 20.0)))

        # 6. Historical Pattern Score (0-100)
        pattern_pos_prob = analogue_data.get("forward_stats", {}).get("10d", {}).get("positive_prob_pct", 50.0)
        pattern_score = pattern_pos_prob

        # Model Score
        model_score = round((tech_score + macro_score + pattern_score) / 3.0, 1)

        # Composite Calculation
        composite = (
            tech_score * weights.get("technical", 0.25) +
            macro_score * weights.get("macro", 0.20) +
            mom_score * weights.get("momentum", 0.15) +
            vol_score * weights.get("volatility", 0.10) +
            rv_score * weights.get("relative_value", 0.15) +
            pattern_score * weights.get("historical_pattern", 0.15)
        )

        return {
            "composite_score": round(composite, 1),
            "sub_scores": {
                "technical_score": round(tech_score, 1),
                "macro_score": round(macro_score, 1),
                "momentum_score": round(mom_score, 1),
                "volatility_score": round(vol_score, 1),
                "relative_value_score": round(rv_score, 1),
                "historical_pattern_score": round(pattern_score, 1),
                "model_score": round(model_score, 1)
            },
            "weights": weights
        }
