import numpy as np
import pandas as pd

class ScoreEngine:
    """Calculates transparent 0-100 scores across technical, macro, momentum, volatility, relative value, pattern, and composite dimensions."""

    def __init__(self, weights: dict = None):
        self.weights = weights or {
            "technical": 0.25,
            "macro": 0.20,
            "momentum": 0.15,
            "volatility": 0.10,
            "relative_value": 0.15,
            "pattern": 0.15
        }

    def compute_scores(
        self,
        tech_row: pd.Series,
        macro_dict: dict,
        ratio_dict: dict,
        analogue_dict: dict,
        model_prob: float = 0.50
    ) -> dict:
        """
        Computes transparent 0-100 scores and weighted composite score for a metal.
        0 = Strongly Bearish / Risk Negative, 50 = Neutral, 100 = Strongly Bullish / Supportive.
        """
        # 1. Technical Score (0-100)
        t_score = 50.0
        if "dist_sma_20" in tech_row and not pd.isna(tech_row["dist_sma_20"]):
            t_score += np.clip(tech_row["dist_sma_20"] * 5, -20, 20)
        if "dist_sma_50" in tech_row and not pd.isna(tech_row["dist_sma_50"]):
            t_score += np.clip(tech_row["dist_sma_50"] * 3, -15, 15)
        if tech_row.get("breakout_20d", False):
            t_score += 15.0
        if tech_row.get("breakdown_20d", False):
            t_score -= 15.0
        tech_score = float(np.clip(t_score, 0, 100))

        # 2. Momentum Score (0-100)
        rsi = tech_row.get("rsi_14", 50.0) or 50.0
        macd_hist = tech_row.get("macd_hist", 0.0) or 0.0
        m_score = 50.0 + (rsi - 50.0) * 0.8 + np.clip(macd_hist * 10, -15, 15)
        momentum_score = float(np.clip(m_score, 0, 100))

        # 3. Volatility Score (0-100)
        # Low volatility / stable expansion = higher score for trend continuation
        vol_20d = tech_row.get("volatility_20d", 0.15) or 0.15
        v_score = 100.0 - np.clip(vol_20d * 200, 0, 80)
        volatility_score = float(np.clip(v_score, 0, 100))

        # 4. Macro Score (0-100)
        p_regime = macro_dict.get("primary_regime", "Neutral")
        if p_regime in ["Risk-Off", "Stagflationary", "Disinflationary Easing"]:
            macro_score = 75.0
        elif p_regime in ["Tightening USD Bull"]:
            macro_score = 25.0
        elif p_regime == "Inflationary":
            macro_score = 65.0
        else:
            macro_score = 50.0

        # 5. Relative Value Score (0-100)
        zscore = ratio_dict.get("ratio_zscore", 0.0) or 0.0
        # If Gold/Silver ratio is very high (zscore > 1.5), relative value favors Silver
        rv_score = 50.0 - (zscore * 15.0)
        relative_value_score = float(np.clip(rv_score, 0, 100))

        # 6. Historical Pattern Score (0-100)
        win_rate = analogue_dict.get("forward_statistics", {}).get("10d", {}).get("win_rate_pct", 50.0)
        pattern_score = float(np.clip(win_rate, 0, 100))

        # 7. Model Score (0-100)
        model_score = float(np.clip(model_prob * 100.0, 0, 100))

        # Composite Score Calculation
        composite = (
            tech_score * self.weights.get("technical", 0.25) +
            macro_score * self.weights.get("macro", 0.20) +
            momentum_score * self.weights.get("momentum", 0.15) +
            volatility_score * self.weights.get("volatility", 0.10) +
            relative_value_score * self.weights.get("relative_value", 0.15) +
            pattern_score * self.weights.get("pattern", 0.15)
        )

        return {
            "technical_score": round(tech_score, 1),
            "macro_score": round(macro_score, 1),
            "momentum_score": round(momentum_score, 1),
            "volatility_score": round(volatility_score, 1),
            "relative_value_score": round(relative_value_score, 1),
            "historical_pattern_score": round(pattern_score, 1),
            "model_score": round(model_score, 1),
            "composite_score": round(composite, 1),
            "score_interpretation": "Bullish" if composite > 60 else ("Bearish" if composite < 40 else "Neutral")
        }
