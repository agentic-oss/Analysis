import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

class TransparentScorer:
    """
    Transparent composite scoring model (0-100 scale):
    - Technical Score (0-100)
    - Macro Score (0-100)
    - Momentum Score (0-100)
    - Volatility Score (0-100)
    - Relative Value Score (0-100)
    - Historical Pattern Score (0-100)

    Composite Score = sum(weight_i * score_i)
    Stores all component scores and rationales.
    """
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "technical": 0.25,
            "macro": 0.20,
            "momentum": 0.15,
            "volatility": 0.10,
            "relative_value": 0.15,
            "pattern": 0.15
        }

    def compute_composite_score(
        self,
        row: pd.Series,
        macro_summary: Dict[str, Any],
        ratio_summary: Dict[str, Any],
        analogue_summary: Dict[str, Any]
    ) -> Dict[str, Any]:

        # 1. Technical Score
        close = row.get("close", 0)
        sma20 = row.get("sma_20", close)
        sma50 = row.get("sma_50", close)
        sma200 = row.get("sma_200", close)

        tech_points = 50.0
        if close > sma20: tech_points += 15.0
        if close > sma50: tech_points += 15.0
        if close > sma200: tech_points += 20.0
        tech_score = float(np.clip(tech_points, 0.0, 100.0))

        # 2. Macro Score
        macro_points = 50.0
        dxy_corr = macro_summary.get("DXY", {}).get("60d", 0.0)
        if dxy_corr is not None and dxy_corr < -0.3:
            macro_points += 15.0 # Normal gold inversely correlated with USD
        vix_level = macro_summary.get("VIX_level", macro_summary.get("inputs", {}).get("vix_level", 15.0))
        if vix_level > 20.0:
            macro_points += 15.0 # Risk-off supportive for metals
        macro_score = float(np.clip(macro_points, 0.0, 100.0))

        # 3. Momentum Score
        rsi = row.get("rsi_14", 50.0)
        macd_hist = row.get("macd_histogram", 0.0)
        mom_points = 50.0
        if 50.0 <= rsi <= 70.0:
            mom_points += 25.0
        elif rsi > 70.0:
            mom_points += 10.0 # Overbought caution
        elif rsi < 30.0:
            mom_points -= 20.0 # Oversold weakness
        if macd_hist > 0:
            mom_points += 25.0
        mom_score = float(np.clip(mom_points, 0.0, 100.0))

        # 4. Volatility Score
        vol_20d = row.get("volatility_20d", 15.0)
        vol_points = 50.0
        if 10.0 <= vol_20d <= 20.0:
            vol_points += 30.0 # Stable volatility
        elif vol_20d > 25.0:
            vol_points -= 20.0 # High volatility risk
        vol_score = float(np.clip(vol_points, 0.0, 100.0))

        # 5. Relative Value Score
        rv_points = 50.0
        gsr_z = ratio_summary.get("current_zscore", 0.0)
        if gsr_z is not None:
            if gsr_z > 1.5: # Gold expensive relative to Silver -> Silver favored, Gold slightly penalized
                rv_points -= 10.0
            elif gsr_z < -1.5:
                rv_points += 20.0
        rv_score = float(np.clip(rv_points, 0.0, 100.0))

        # 6. Pattern / Analogue Score
        fwd_5d_pos = analogue_summary.get("forward_statistics", {}).get("5d", {}).get("positive_probability_pct", 50.0)
        pattern_score = float(np.clip(fwd_5d_pos, 0.0, 100.0))

        # Weighted Composite
        composite = (
            self.weights["technical"] * tech_score +
            self.weights["macro"] * macro_score +
            self.weights["momentum"] * mom_score +
            self.weights["volatility"] * vol_score +
            self.weights["relative_value"] * rv_score +
            self.weights["pattern"] * pattern_score
        )

        return {
            "composite_score": round(float(composite), 2),
            "component_scores": {
                "technical": round(tech_score, 2),
                "macro": round(macro_score, 2),
                "momentum": round(mom_score, 2),
                "volatility": round(vol_score, 2),
                "relative_value": round(rv_score, 2),
                "pattern": round(pattern_score, 2)
            },
            "weights": self.weights
        }
