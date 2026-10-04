import pandas as pd
import numpy as np
from typing import Dict, Any

class MarketScorer:
    """Calculates transparent sub-scores (0-100) and an explainable composite score."""

    @staticmethod
    def _safe_float(val: Any, default: float = 0.0) -> float:
        if val is None or pd.isna(val):
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    @classmethod
    def calculate_scores(
        cls,
        latest_feature_row: pd.Series,
        macro_dict: Dict[str, Any],
        analogue_dict: Dict[str, Any],
        weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Compute 0-100 scores for Technical, Macro, Momentum, Volatility, Relative Value, Analogue, and Model."""

        # 1. Technical Score (0-100): MAs, RSI, MACD
        tech_score = 50.0
        rsi = cls._safe_float(latest_feature_row.get("rsi_14"), 50.0)
        dist_200 = cls._safe_float(latest_feature_row.get("dist_sma_200"), 0.0)
        macd = cls._safe_float(latest_feature_row.get("macd"), 0.0)

        if rsi > 50:
            tech_score += min((rsi - 50) * 0.8, 20.0)
        else:
            tech_score -= min((50 - rsi) * 0.8, 20.0)

        if dist_200 > 0:
            tech_score += min(dist_200 * 100, 15.0)
        else:
            tech_score -= min(abs(dist_200) * 100, 15.0)

        if macd > 0:
            tech_score += 10.0
        else:
            tech_score -= 10.0

        tech_score = float(np.clip(tech_score, 0.0, 100.0))

        # 2. Macro Score (0-100): Rates, DXY, VIX
        macro_score = 50.0
        components = macro_dict.get("explainable_components", {})
        dxy_ret = cls._safe_float(components.get("dxy_return_20d"), 0.0)
        rates_change = cls._safe_float(components.get("us10y_change_20d"), 0.0)

        # Falling DXY and falling yields are bullish for precious metals
        if dxy_ret < 0:
            macro_score += min(abs(dxy_ret) * 5.0, 20.0)
        else:
            macro_score -= min(dxy_ret * 5.0, 20.0)

        if rates_change < 0:
            macro_score += min(abs(rates_change) * 50.0, 20.0)
        else:
            macro_score -= min(rates_change * 50.0, 20.0)

        macro_score = float(np.clip(macro_score, 0.0, 100.0))

        # 3. Momentum Score (0-100)
        ret_20d = cls._safe_float(latest_feature_row.get("return_20d"), 0.0)
        momentum_score = float(np.clip(50.0 + (ret_20d * 200.0), 0.0, 100.0))

        # 4. Volatility Score (0-100): Lower volatility = higher stability score
        vol_20d = cls._safe_float(latest_feature_row.get("volatility_20d"), 0.15)
        volatility_score = float(np.clip(100.0 - (vol_20d * 250.0), 0.0, 100.0))

        # 5. Relative Value Score (0-100): Gold/Silver ratio z-score
        ratio_z = cls._safe_float(latest_feature_row.get("gold_silver_ratio_zscore"), 0.0)
        rel_value_score = float(np.clip(50.0 + (ratio_z * 15.0), 0.0, 100.0))

        # 6. Analogue Score (0-100): Based on forward probability of positive return
        stats = analogue_dict.get("forward_statistics", {}).get("5d", {})
        prob_pos = cls._safe_float(stats.get("prob_positive"), 0.5)
        analogue_score = float(np.clip(prob_pos * 100.0, 0.0, 100.0))

        # 7. Model Score (0-100)
        model_score = 50.0

        sub_scores = {
            "technical_score": round(tech_score, 2),
            "macro_score": round(macro_score, 2),
            "momentum_score": round(momentum_score, 2),
            "volatility_score": round(volatility_score, 2),
            "relative_value_score": round(rel_value_score, 2),
            "analogue_score": round(analogue_score, 2),
            "model_score": round(model_score, 2)
        }

        default_w = {
            "technical": 0.25,
            "macro": 0.20,
            "momentum": 0.15,
            "volatility": 0.10,
            "relative_value": 0.10,
            "analogue": 0.10,
            "model": 0.10
        }
        actual_weights = {**default_w, **weights}

        composite = (
            sub_scores["technical_score"] * actual_weights.get("technical", 0.25) +
            sub_scores["macro_score"] * actual_weights.get("macro", 0.20) +
            sub_scores["momentum_score"] * actual_weights.get("momentum", 0.15) +
            sub_scores["volatility_score"] * actual_weights.get("volatility", 0.10) +
            sub_scores["relative_value_score"] * actual_weights.get("relative_value", 0.10) +
            sub_scores["analogue_score"] * actual_weights.get("analogue", 0.10) +
            sub_scores["model_score"] * actual_weights.get("model", 0.10)
        )

        return {
            "composite_score": round(composite, 2),
            "sub_scores": sub_scores,
            "weights": actual_weights,
            "explanation": f"Composite score {round(composite, 2)} derived from Technical={sub_scores['technical_score']}, Macro={sub_scores['macro_score']}, Momentum={sub_scores['momentum_score']}."
        }
