from typing import Dict, Any, List, Optional
import numpy as np


class ScoringEngine:
    """Calculates transparent 0-100 component scores and composite score with explicit confidence rationale."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "technical": 0.25,
            "macro": 0.25,
            "momentum": 0.15,
            "volatility": 0.10,
            "relative_value": 0.15,
            "pattern": 0.10,
        }

    def compute_all_scores(
        self,
        technical_metrics: Dict[str, Any],
        macro_metrics: Dict[str, Any],
        relative_value_metrics: Dict[str, Any],
        analogue_metrics: Dict[str, Any],
        model_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:

        # Technical Score (0 to 100)
        dist_200 = technical_metrics.get("dist_sma_200", 0.0)
        dist_50 = technical_metrics.get("dist_sma_50", 0.0)
        tech_base = 50.0 + (dist_50 * 200.0) + (dist_200 * 100.0)
        technical_score = float(np.clip(tech_base, 0, 100))

        # Momentum Score (0 to 100)
        rsi = technical_metrics.get("rsi_14", 50.0)
        macd_hist = technical_metrics.get("macd_hist", 0.0)
        mom_base = rsi + (macd_hist * 10.0)
        momentum_score = float(np.clip(mom_base, 0, 100))

        # Volatility Score (0 to 100)
        vol_20 = technical_metrics.get("volatility_20d", 0.15)
        # Lower volatility = higher stability score
        vol_score = float(np.clip(100.0 - (vol_20 * 250.0), 0, 100))

        # Macro Score (0 to 100)
        sentiment = macro_metrics.get("sentiment", "Neutral")
        dollar_stance = macro_metrics.get("dollar_stance", "Dollar Neutral")
        macro_base = 50.0
        if sentiment == "Risk-Off":
            macro_base += 15.0  # Safe haven demand for gold
        if dollar_stance == "Dollar Weak":
            macro_base += 15.0
        elif dollar_stance == "Dollar Strong":
            macro_base -= 15.0
        macro_score = float(np.clip(macro_base, 0, 100))

        # Relative Value Score (0 to 100)
        zscore = relative_value_metrics.get("gs_ratio_zscore", 0.0)
        # Z-score mean reversion signal
        rv_base = 50.0 - (zscore * 15.0)
        relative_value_score = float(np.clip(rv_base, 0, 100))

        # Historical Pattern Score (0 to 100)
        stats_5d = analogue_metrics.get("forward_statistics", {}).get("5d", {})
        pos_prob = stats_5d.get("prob_positive", 0.5)
        pattern_score = float(np.clip(pos_prob * 100.0, 0, 100))

        # Model Score (0 to 100)
        mod_prob = model_metrics.get("prob_positive", 0.5)
        model_score = float(np.clip(mod_prob * 100.0, 0, 100))

        # Composite Score
        composite = (
            (technical_score * self.weights["technical"])
            + (macro_score * self.weights["macro"])
            + (momentum_score * self.weights["momentum"])
            + (vol_score * self.weights["volatility"])
            + (relative_value_score * self.weights["relative_value"])
            + (pattern_score * self.weights["pattern"])
        )

        # Confidence Score (0-100) & Reasons
        sample_count = stats_5d.get("sample_count", 0)
        model_agreement = abs(mod_prob - 0.5) * 2.0

        confidence_val = int(min(100, 40 + (min(sample_count, 30) * 1.0) + (model_agreement * 20.0)))

        reasons = [
            f"{int(pos_prob * 100)}% positive forward return frequency among historical analogues (samples={sample_count})",
            f"Technical structure score: {round(technical_score, 1)}/100",
            f"Macro environment score: {round(macro_score, 1)}/100",
            f"Model consensus probability: {round(mod_prob * 100, 1)}%",
        ]

        return {
            "technical_score": round(technical_score, 2),
            "macro_score": round(macro_score, 2),
            "momentum_score": round(momentum_score, 2),
            "volatility_score": round(vol_score, 2),
            "relative_value_score": round(relative_value_score, 2),
            "pattern_score": round(pattern_score, 2),
            "model_score": round(model_score, 2),
            "composite_score": round(composite, 2),
            "confidence_score": confidence_val,
            "confidence_reasons": reasons,
        }
