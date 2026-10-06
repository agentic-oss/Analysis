import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class TransparentScoringSystem:
    """Calculates multi-factor transparent scores (0-100) and confidence frameworks with explainable reasons."""

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            "technical": 0.25,
            "macro": 0.20,
            "momentum": 0.15,
            "volatility": 0.10,
            "relative_value": 0.15,
            "historical_pattern": 0.15
        }

    def compute_scores(self, tech_metrics: Dict[str, Any], macro_info: Dict[str, Any], ratio_info: Dict[str, Any], analogue_info: Dict[str, Any], model_info: Dict[str, Any]) -> Dict[str, Any]:
        # 1. Technical Score (0-100)
        dist_200 = tech_metrics.get("dist_sma_200", 0.0)
        tech_score = max(0.0, min(100.0, 50.0 + dist_200 * 250.0))

        # 2. Momentum Score (0-100)
        rsi = tech_metrics.get("rsi_14", 50.0)
        momentum_score = max(0.0, min(100.0, rsi))

        # 3. Volatility Score (0-100, higher score = low/stable volatility)
        vol_20 = tech_metrics.get("volatility_20d", 0.15)
        vol_score = max(0.0, min(100.0, 100.0 - (vol_20 * 200.0)))

        # 4. Relative Value Score (0-100)
        zscore = ratio_info.get("ratio_zscore", 0.0)
        rv_score = max(0.0, min(100.0, 50.0 - zscore * 20.0))

        # 5. Macro Score (0-100)
        p_regime = macro_info.get("primary_regime", "Neutral")
        macro_map = {
            "Risk-Off": 80.0, "Inflationary": 85.0, "Stagflationary": 90.0,
            "Easing": 75.0, "Neutral": 50.0, "Risk-On": 40.0, "Tightening": 35.0, "Disinflationary": 30.0
        }
        macro_score = macro_map.get(p_regime, 50.0)

        # 6. Historical Pattern Score (0-100)
        f_stats = analogue_info.get("forward_stats", {}).get("5d", {})
        prob_pos = f_stats.get("prob_positive", 0.5)
        pattern_score = float(prob_pos * 100.0)

        # Composite Score (Weighted Average)
        composite = (
            tech_score * self.weights.get("technical", 0.25) +
            macro_score * self.weights.get("macro", 0.20) +
            momentum_score * self.weights.get("momentum", 0.15) +
            vol_score * self.weights.get("volatility", 0.10) +
            rv_score * self.weights.get("relative_value", 0.15) +
            pattern_score * self.weights.get("historical_pattern", 0.15)
        )

        # Confidence Calculation
        reasons = []
        confidence_val = 50.0

        if prob_pos >= 0.60:
            confidence_val += 15.0
            reasons.append(f"{prob_pos * 100:.0f}% positive forward return frequency among historical analogues")

        if rsi > 55 and tech_score > 60:
            confidence_val += 10.0
            reasons.append("Technical trend and momentum indicators aligned")

        if p_regime in ["Inflationary", "Risk-Off", "Stagflationary"]:
            confidence_val += 10.0
            reasons.append(f"Macro regime ({p_regime}) supportive")

        m_agree = model_info.get("model_agreement", "0/0")
        if "/" in m_agree:
            parts = m_agree.split("/")
            if int(parts[0]) >= int(parts[1]) * 0.75:
                confidence_val += 15.0
                reasons.append(f"Model agreement high ({m_agree})")

        confidence_val = max(0.0, min(100.0, confidence_val))

        return {
            "composite_score": float(composite),
            "confidence_score": float(confidence_val),
            "confidence_reasons": reasons,
            "components": {
                "technical_score": float(tech_score),
                "macro_score": float(macro_score),
                "momentum_score": float(momentum_score),
                "volatility_score": float(vol_score),
                "relative_value_score": float(rv_score),
                "historical_pattern_score": float(pattern_score)
            }
        }
