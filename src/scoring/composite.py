"""
Composite scoring framework calculating sub-scores (0-100) and weighted final score.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


class CompositeScorer:
    """
    Transparent multi-factor scoring engine:
    - Technical Score (0-100)
    - Macro Score (0-100)
    - Momentum Score (0-100)
    - Volatility Score (0-100)
    - Relative Value Score (0-100)
    - Historical Pattern Score (0-100)
    """

    def __init__(self, weights: Dict[str, float] = None):
        if weights is None:
            self.weights = {
                "technical": 0.25,
                "macro": 0.20,
                "momentum": 0.15,
                "volatility": 0.10,
                "relative_value": 0.15,
                "historical_pattern": 0.15,
            }
        else:
            self.weights = weights

    def calculate_scores(self, feature_row: pd.Series, analogue_stats: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Calculates individual component scores (0-100) and final composite score.
        50 = Neutral, >50 = Bullish/Positive, <50 = Bearish/Negative.
        """
        # Technical Score
        tech_score = 50.0
        rsi = feature_row.get("rsi_14", 50.0)
        dist_200 = feature_row.get("dist_sma_200_pct", 0.0)
        if not pd.isna(rsi):
            tech_score += (rsi - 50.0) * 0.5
        if not pd.isna(dist_200):
            tech_score += np.clip(dist_200 * 100.0, -15.0, 15.0)
        tech_score = float(np.clip(tech_score, 0.0, 100.0))

        # Momentum Score
        mom_score = 50.0
        macd_hist = feature_row.get("macd_histogram", 0.0)
        ret_5d = feature_row.get("gold_return_1d", 0.0)
        if not pd.isna(macd_hist):
            mom_score += np.clip(macd_hist * 10.0, -20.0, 20.0)
        if not pd.isna(ret_5d):
            mom_score += np.clip(ret_5d * 200.0, -20.0, 20.0)
        mom_score = float(np.clip(mom_score, 0.0, 100.0))

        # Macro Score
        macro_score = 50.0
        dxy_ret = feature_row.get("dxy_return_20d", 0.0)
        us10y_chg = feature_row.get("us10y_change_20d", 0.0)
        if not pd.isna(dxy_ret):
            macro_score -= np.clip(dxy_ret * 300.0, -25.0, 25.0)  # Strong DXY usually negative for Gold
        if not pd.isna(us10y_chg):
            macro_score -= np.clip(us10y_chg * 10.0, -15.0, 15.0)
        macro_score = float(np.clip(macro_score, 0.0, 100.0))

        # Volatility Score
        vol_score = 50.0
        vol_20d = feature_row.get("volatility_20d", 0.15)
        if not pd.isna(vol_20d):
            if vol_20d > 0.25:
                vol_score = 30.0  # Extreme volatility is risky/negative
            elif vol_20d < 0.10:
                vol_score = 65.0  # Low volatility consolidation
            else:
                vol_score = 50.0

        # Relative Value Score
        rv_score = 50.0
        gs_zscore = feature_row.get("gold_silver_ratio_zscore", 0.0)
        if not pd.isna(gs_zscore):
            # High ratio z-score means Gold is expensive relative to Silver
            rv_score -= np.clip(gs_zscore * 15.0, -30.0, 30.0)
        rv_score = float(np.clip(rv_score, 0.0, 100.0))

        # Historical Pattern Score
        pattern_score = 50.0
        if analogue_stats and "5d" in analogue_stats and "positive_probability_pct" in analogue_stats["5d"]:
            pos_prob = analogue_stats["5d"]["positive_probability_pct"]
            pattern_score = float(pos_prob)

        # Composite Score Calculation
        composite = (
            self.weights["technical"] * tech_score +
            self.weights["macro"] * macro_score +
            self.weights["momentum"] * mom_score +
            self.weights["volatility"] * vol_score +
            self.weights["relative_value"] * rv_score +
            self.weights["historical_pattern"] * pattern_score
        )
        composite = float(np.clip(composite, 0.0, 100.0))

        return {
            "composite_score": round(composite, 1),
            "sub_scores": {
                "technical": round(tech_score, 1),
                "macro": round(macro_score, 1),
                "momentum": round(mom_score, 1),
                "volatility": round(vol_score, 1),
                "relative_value": round(rv_score, 1),
                "historical_pattern": round(pattern_score, 1),
            },
            "weights": self.weights,
        }
