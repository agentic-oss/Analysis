"""Transparent Multi-Factor Scoring System (0-100 scale)."""

from typing import Dict, Any
import numpy as np
import pandas as pd


class CompositeScorer:
    """Calculates component-wise scores and weighted composite research score (0-100)."""

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            "technical": 0.25,
            "macro": 0.20,
            "momentum": 0.15,
            "volatility": 0.10,
            "relative_value": 0.15,
            "historical_pattern": 0.15
        }

    def compute_scores(
        self,
        tech_indicators: Dict[str, Any],
        macro_correlations: Dict[str, Any],
        ratio_metrics: Dict[str, Any],
        analogue_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculates individual scores (0-100) and final composite score."""

        # 1. Technical Score
        rsi = tech_indicators.get('rsi_14', 50.0)
        dist_200 = tech_indicators.get('distance_sma_200', 0.0)
        dist_50 = tech_indicators.get('distance_sma_50', 0.0)

        tech_score = 50.0
        if dist_50 > 0 and dist_200 > 0:
            tech_score += 20.0
        elif dist_50 < 0 and dist_200 < 0:
            tech_score -= 20.0

        if 40 <= rsi <= 65:
            tech_score += 15.0
        elif rsi > 70:
            tech_score -= 10.0 # Overbought
        elif rsi < 30:
            tech_score += 10.0 # Oversold bounce potential

        tech_score = float(np.clip(tech_score, 0, 100))

        # 2. Momentum Score
        macd_hist = tech_indicators.get('macd_hist', 0.0)
        roc_10 = tech_indicators.get('roc_10', 0.0)

        mom_score = 50.0 + (10.0 if macd_hist > 0 else -10.0) + (15.0 if roc_10 > 0 else -15.0)
        mom_score = float(np.clip(mom_score, 0, 100))

        # 3. Macro Score
        # Supportive macro for gold: weak dollar (negative DXY corr or drop), falling real yields
        dxy_corr = macro_correlations.get('DXY', {}).get('corr_60d', -0.5)
        dxy_ret = macro_correlations.get('DXY', {}).get('return_1d', 0.0)

        macro_score = 50.0
        if dxy_ret < 0:
            macro_score += 15.0
        else:
            macro_score -= 15.0
        macro_score = float(np.clip(macro_score, 0, 100))

        # 4. Volatility Score
        vol = tech_indicators.get('volatility_20d', 0.15)
        vol_score = 70.0 if vol < 0.15 else (50.0 if vol < 0.25 else 30.0)

        # 5. Relative Value Score (Gold/Silver ratio)
        z_score = ratio_metrics.get('z_score_252d', 0.0)
        # Low z-score means Gold is cheap relative to Silver; high z-score means Gold is expensive
        rv_score = 50.0 - (z_score * 15.0)
        rv_score = float(np.clip(rv_score, 0, 100))

        # 6. Historical Pattern Score
        fwd_stats = analogue_stats.get('forward_stats', {}).get('10d', {})
        pos_prob = fwd_stats.get('positive_probability', 0.5)
        pattern_score = float(pos_prob * 100.0)

        # Composite Score Calculation
        composite = (
            self.weights['technical'] * tech_score +
            self.weights['macro'] * macro_score +
            self.weights['momentum'] * mom_score +
            self.weights['volatility'] * vol_score +
            self.weights['relative_value'] * rv_score +
            self.weights['historical_pattern'] * pattern_score
        )

        return {
            "composite_score": round(float(composite), 2),
            "components": {
                "technical_score": round(tech_score, 2),
                "macro_score": round(macro_score, 2),
                "momentum_score": round(mom_score, 2),
                "volatility_score": round(vol_score, 2),
                "relative_value_score": round(rv_score, 2),
                "historical_pattern_score": round(pattern_score, 2)
            },
            "weights": self.weights
        }
