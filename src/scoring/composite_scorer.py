import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class CompositeScorer:
    """
    Transparent, multi-factor score aggregator (0 to 100 scale).
    Combines:
    - Technical Score (0-100)
    - Macro Score (0-100)
    - Momentum Score (0-100)
    - Volatility Score (0-100)
    - Relative Value Score (0-100)
    - Historical Pattern Score (0-100)
    - Model Score (0-100)
    Calculates a weighted composite research score.
    """

    @staticmethod
    def calculate_scores(tech_row: pd.Series, macro_regime: Dict[str, Any], ratio_metrics: Dict[str, Any], analogue_stats: Dict[str, Any], model_prob: float = 0.5) -> Dict[str, Any]:

        # 1. Technical Score (0-100)
        close = tech_row.get("Close", 100)
        sma_20 = tech_row.get("sma_20", close)
        sma_50 = tech_row.get("sma_50", close)
        sma_200 = tech_row.get("sma_200", close)

        tech_pts = 50.0
        if close > sma_20: tech_pts += 15.0
        if close > sma_50: tech_pts += 15.0
        if close > sma_200: tech_pts += 20.0
        if close < sma_20: tech_pts -= 15.0
        if close < sma_50: tech_pts -= 15.0
        if close < sma_200: tech_pts -= 20.0
        tech_score = float(np.clip(tech_pts, 0.0, 100.0))

        # 2. Momentum Score (0-100)
        rsi = tech_row.get("rsi_14", 50.0)
        macd = tech_row.get("macd", 0.0)
        macd_signal = tech_row.get("macd_signal", 0.0)

        mom_pts = 50.0
        if 50.0 <= rsi < 70.0: mom_pts += 20.0
        elif rsi >= 70.0: mom_pts += 10.0 # Extreme overbought
        elif 30.0 < rsi < 50.0: mom_pts -= 15.0
        elif rsi <= 30.0: mom_pts += 15.0 # Oversold potential rebound

        if macd > macd_signal: mom_pts += 15.0
        else: mom_pts -= 15.0
        momentum_score = float(np.clip(mom_pts, 0.0, 100.0))

        # 3. Macro Score (0-100)
        prim_macro = macro_regime.get("primary_macro_regime", "Disinflationary")
        risk_regime = macro_regime.get("risk_regime", "Neutral Risk")
        monetary = macro_regime.get("monetary_regime", "Neutral Policy")

        macro_pts = 50.0
        if prim_macro == "Inflationary" or prim_macro == "Stagflationary": macro_pts += 20.0
        if risk_regime == "Risk-Off": macro_pts += 15.0 # Flight to safe haven gold
        if monetary == "Easing": macro_pts += 15.0 # Easing lowers real rate, favors gold
        if monetary == "Tightening": macro_pts -= 15.0
        macro_score = float(np.clip(macro_pts, 0.0, 100.0))

        # 4. Volatility Score (0-100)
        vol_20d = tech_row.get("volatility_20d", 0.15)
        if vol_20d < 0.12: vol_score = 70.0 # Low vol consolidation preceding expansion
        elif vol_20d > 0.25: vol_score = 30.0 # High vol turbulent
        else: vol_score = 50.0

        # 5. Relative Value Score (0-100)
        gsr_z = ratio_metrics.get("z_score", 0.0)
        if gsr_z > 1.5: rel_val_score = 75.0 # Favor silver / bullish metals mean reversion
        elif gsr_z < -1.5: rel_val_score = 35.0
        else: rel_val_score = 50.0

        # 6. Historical Pattern Score (0-100)
        fwd_5d = analogue_stats.get("5d", {})
        if fwd_5d and "prob_positive_pct" in fwd_5d:
            hist_pattern_score = float(fwd_5d["prob_positive_pct"])
        else:
            hist_pattern_score = 50.0

        # 7. Model Score (0-100)
        model_score = float(model_prob * 100.0)

        # Composite Score (Weighted Average)
        composite = (
            0.20 * tech_score +
            0.20 * macro_score +
            0.15 * momentum_score +
            0.10 * vol_score +
            0.10 * rel_val_score +
            0.15 * hist_pattern_score +
            0.10 * model_score
        )

        return {
            "composite_score": round(float(composite), 2),
            "technical_score": round(tech_score, 2),
            "macro_score": round(macro_score, 2),
            "momentum_score": round(momentum_score, 2),
            "volatility_score": round(vol_score, 2),
            "relative_value_score": round(rel_val_score, 2),
            "historical_pattern_score": round(hist_pattern_score, 2),
            "model_score": round(model_score, 2)
        }
