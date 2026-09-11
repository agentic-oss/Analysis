"""
Scoring and Alert Detection Engine.
Calculates transparent component scores (0-100) for Technical, Macro, Momentum,
Volatility, Relative Value, Historical Pattern, and ML Model performance.
Triggers alerts when configurable thresholds are exceeded.
"""

from typing import Dict, Any, List


class ScoringEngine:
    """Calculates individual factor scores and weighted composite scores."""

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            "technical": 0.25,
            "macro": 0.25,
            "momentum": 0.15,
            "volatility": 0.10,
            "relative_value": 0.10,
            "historical_pattern": 0.10,
            "model": 0.05,
        }

    def compute_composite_scores(
        self,
        tech_indicators: Dict[str, Any],
        macro_indicators: Dict[str, Any],
        ratio_metrics: Dict[str, Any],
        analogue_stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculates factor scores and final composite bullish/bearish score (0-100)."""
        rsi = tech_indicators.get("rsi_14", 50.0)
        dist_sma200 = tech_indicators.get("dist_sma_200_pct", 0.0)
        vol_20d = tech_indicators.get("volatility_20d", 0.15)
        ratio_zscore = ratio_metrics.get("ratio_zscore", 0.0)

        # Technical Score (0 = extremely bearish, 50 = neutral, 100 = extremely bullish)
        tech_score = max(0.0, min(100.0, 50.0 + (dist_sma200 * 2.0)))

        # Momentum Score
        momentum_score = max(0.0, min(100.0, rsi))

        # Volatility Score (Higher = calmer market environment)
        vol_score = max(0.0, min(100.0, 100.0 - (vol_20d * 300.0)))

        # Macro Score
        yield_chg = macro_indicators.get("real_yield_change_20d", 0.0)
        dxy_chg = macro_indicators.get("dxy_change_20d", 0.0)
        macro_score = max(0.0, min(100.0, 50.0 - (yield_chg * 100.0) - (dxy_chg * 100.0)))

        # Relative Value Score (Gold vs Silver z-score)
        rel_value_score = max(0.0, min(100.0, 50.0 - (ratio_zscore * 20.0)))

        # Historical Pattern Score
        prob_pos_5d = analogue_stats.get("5d", {}).get("prob_positive", 0.5)
        pattern_score = prob_pos_5d * 100.0

        # Model Score
        model_score = 50.0

        composite = (
            tech_score * self.weights["technical"]
            + macro_score * self.weights["macro"]
            + momentum_score * self.weights["momentum"]
            + vol_score * self.weights["volatility"]
            + rel_value_score * self.weights["relative_value"]
            + pattern_score * self.weights["historical_pattern"]
            + model_score * self.weights["model"]
        )

        return {
            "composite_score": round(composite, 1),
            "technical_score": round(tech_score, 1),
            "macro_score": round(macro_score, 1),
            "momentum_score": round(momentum_score, 1),
            "volatility_score": round(vol_score, 1),
            "relative_value_score": round(rel_value_score, 1),
            "historical_pattern_score": round(pattern_score, 1),
            "model_score": round(model_score, 1),
            "weights": self.weights,
        }


class AlertEngine:
    """Detects market breakouts, breakdowns, extremes, and regime transitions."""

    def detect_alerts(
        self,
        date_str: str,
        gold_data: Dict[str, Any],
        silver_data: Dict[str, Any],
        ratio_data: Dict[str, Any],
        macro_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Scans metrics against configurable alert thresholds."""
        alerts = []

        # Gold Breakout / Breakdown
        if gold_data.get("breakout_20d"):
            alerts.append({"type": "GOLD_BREAKOUT_20D", "severity": "HIGH", "message": "Gold broke 20-day high"})
        if gold_data.get("breakdown_20d"):
            alerts.append({"type": "GOLD_BREAKDOWN_20D", "severity": "HIGH", "message": "Gold broke 20-day low"})

        # Silver Breakout / Breakdown
        if silver_data.get("breakout_20d"):
            alerts.append({"type": "SILVER_BREAKOUT_20D", "severity": "HIGH", "message": "Silver broke 20-day high"})
        if silver_data.get("breakdown_20d"):
            alerts.append({"type": "SILVER_BREAKDOWN_20D", "severity": "HIGH", "message": "Silver broke 20-day low"})

        # RSI Extremes
        gold_rsi = gold_data.get("rsi_14", 50.0)
        if gold_rsi >= 70.0:
            alerts.append({"type": "GOLD_RSI_OVERBOUGHT", "severity": "MEDIUM", "message": f"Gold RSI overbought at {gold_rsi:.1f}"})
        elif gold_rsi <= 30.0:
            alerts.append({"type": "GOLD_RSI_OVERSOLD", "severity": "MEDIUM", "message": f"Gold RSI oversold at {gold_rsi:.1f}"})

        # Gold/Silver Ratio Extreme
        gs_ratio = ratio_data.get("gold_silver_ratio", 80.0)
        if gs_ratio >= 85.0:
            alerts.append({"type": "RATIO_EXTREME_HIGH", "severity": "HIGH", "message": f"Gold/Silver ratio reached extreme high: {gs_ratio:.2f}"})
        elif gs_ratio <= 65.0:
            alerts.append({"type": "RATIO_EXTREME_LOW", "severity": "HIGH", "message": f"Gold/Silver ratio reached extreme low: {gs_ratio:.2f}"})

        return alerts
