import logging
import numpy as np
import pandas as pd
from typing import Dict, Any

logger = logging.getLogger(__name__)

class TransparentScoringSystem:
    """Calculates 0-100 scores for technicals, macro, momentum, volatility, relative value, pattern, and model."""

    @staticmethod
    def calculate_technical_score(rsi: float, price: float, sma_20: float, sma_50: float, sma_200: float, macd_hist: float) -> float:
        score = 50.0
        if rsi > 50: score += (rsi - 50) * 0.5
        else: score -= (50 - rsi) * 0.5

        if price > sma_20: score += 5
        if price > sma_50: score += 10
        if price > sma_200: score += 15
        if macd_hist > 0: score += 10

        return round(float(np.clip(score, 0.0, 100.0)), 1)

    @staticmethod
    def calculate_macro_score(dxy_return: float, real_yield_change: float, vix: float) -> float:
        score = 50.0
        # DXY negative -> Gold/Silver supportive
        if dxy_return < 0: score += min(abs(dxy_return) * 500, 25)
        else: score -= min(dxy_return * 500, 25)

        # Real yield negative change -> Gold supportive
        if real_yield_change < 0: score += min(abs(real_yield_change) * 50, 25)
        else: score -= min(real_yield_change * 50, 25)

        return round(float(np.clip(score, 0.0, 100.0)), 1)

    @classmethod
    def calculate_composite_score(cls, scores: Dict[str, float], weights: Dict[str, float]) -> Dict[str, Any]:
        composite = 0.0
        total_weight = sum(weights.values())

        for key, weight in weights.items():
            s_val = scores.get(f"{key}_score", 50.0)
            composite += s_val * (weight / total_weight)

        return {
            "composite_score": round(float(composite), 1),
            "score_breakdown": scores
        }
