"""
Multi-Factor Scoring & Alert Generation Engine.
Calculates transparent individual component scores (0-100) and composite score:
Technical, Macro, Momentum, Volatility, Relative Value, Pattern, Model scores.
Generates explicit confidence reasoning and triggers alerts when threshold conditions are met.
Outputs to data/analysis/alerts/YYYY-MM-DD.json.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class ScoringAndAlertEngine:
    """Computes transparent scores, confidence ratings with reasons, and triggers configurable alerts."""

    def __init__(self, alert_thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = alert_thresholds or {
            "price_change_pct_1d": 2.0,
            "rsi_overbought": 70.0,
            "rsi_oversold": 30.0,
            "volatility_spike_zscore": 2.0,
            "gold_silver_ratio_zscore_extreme": 2.0,
            "yield_change_bp_1d": 10.0
        }

    def compute_scores(
        self,
        tech_signals: Dict[str, Any],
        rsi: float = 50.0,
        macd_hist: float = 0.0,
        vol_20d: float = 15.0,
        gs_ratio_zscore: float = 0.0,
        macro_regime: str = "NEUTRAL",
        analogue_win_rate: float = 50.0,
        model_prob_pos: float = 50.0
    ) -> Dict[str, Any]:
        """Calculates transparent sub-scores (0-100) and weighted composite score."""

        # 1. Technical Score (0-100)
        tech_base = 50.0
        if tech_signals.get("golden_cross"): tech_base += 20.0
        if tech_signals.get("death_cross"): tech_base -= 20.0
        if tech_signals.get("breakout_20d"): tech_base += 15.0
        if tech_signals.get("breakdown_20d"): tech_base -= 15.0
        tech_score = float(np.clip(tech_base, 0.0, 100.0))

        # 2. Momentum Score
        mom_base = 50.0
        if rsi > 60.0: mom_base += 15.0
        elif rsi < 40.0: mom_base -= 15.0
        if macd_hist > 0: mom_base += 15.0
        elif macd_hist < 0: mom_base -= 15.0
        momentum_score = float(np.clip(mom_base, 0.0, 100.0))

        # 3. Macro Score
        macro_map = {
            "STAGFLATIONARY": 85.0, "INFLATIONARY": 75.0, "RISK_OFF": 70.0,
            "EASING": 65.0, "NEUTRAL": 50.0, "TIGHTENING": 35.0, "RISK_ON": 40.0, "DEFLATIONARY": 30.0
        }
        macro_score = macro_map.get(macro_regime, 50.0)

        # 4. Volatility Score (Favorable stability vs extreme panic)
        vol_score = 70.0 if (10.0 <= vol_20d <= 22.0) else (40.0 if vol_20d > 28.0 else 50.0)

        # 5. Relative Value Score
        rv_score = float(np.clip(50.0 + (gs_ratio_zscore * 15.0), 0.0, 100.0))

        # 6. Pattern Score (Historical Analogue Win Rate)
        pattern_score = float(np.clip(analogue_win_rate, 0.0, 100.0))

        # 7. Model Score
        model_score = float(np.clip(model_prob_pos, 0.0, 100.0))

        # Composite Score (Weighted average)
        weights = {
            "technical": 0.20,
            "macro": 0.20,
            "momentum": 0.15,
            "volatility": 0.10,
            "relative_value": 0.15,
            "pattern": 0.10,
            "model": 0.10
        }

        composite = (
            tech_score * weights["technical"] +
            macro_score * weights["macro"] +
            momentum_score * weights["momentum"] +
            vol_score * weights["volatility"] +
            rv_score * weights["relative_value"] +
            pattern_score * weights["pattern"] +
            model_score * weights["model"]
        )

        # Confidence framework with explicit reasoning
        reasons = []
        if analogue_win_rate >= 60.0:
            reasons.append(f"{analogue_win_rate:.0f}% positive forward return frequency among historical analogues")
        if tech_signals.get("breakout_20d"):
            reasons.append("20-day price breakout detected")
        if macro_regime in ["INFLATIONARY", "STAGFLATIONARY", "RISK_OFF"]:
            reasons.append(f"Macro regime is supportive ({macro_regime})")
        if abs(model_prob_pos - 50.0) >= 10.0:
            reasons.append(f"Forecasting models agree on directional bias ({model_prob_pos:.0f}%)")

        confidence_score = float(np.clip(50.0 + len(reasons) * 12.0, 30.0, 95.0))

        return {
            "composite_score": float(composite),
            "confidence_score": confidence_score,
            "confidence_reasons": reasons,
            "breakdown": {
                "technical_score": tech_score,
                "macro_score": macro_score,
                "momentum_score": momentum_score,
                "volatility_score": vol_score,
                "relative_value_score": rv_score,
                "pattern_score": pattern_score,
                "model_score": model_score
            }
        }

    def detect_alerts(
        self,
        symbol: str,
        price_change_pct: float,
        rsi: float,
        vol_zscore: float,
        gs_ratio_zscore: float,
        tech_signals: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Triggers actionable market alerts if thresholds are breached."""
        alerts = []

        if abs(price_change_pct) >= self.thresholds["price_change_pct_1d"]:
            alerts.append({
                "type": "LARGE_PRICE_MOVE",
                "severity": "HIGH",
                "symbol": symbol,
                "message": f"Significant 1D move of {price_change_pct:.2f}% detected in {symbol}."
            })

        if rsi >= self.thresholds["rsi_overbought"]:
            alerts.append({
                "type": "RSI_EXTREME_OVERBOUGHT",
                "severity": "MEDIUM",
                "symbol": symbol,
                "message": f"RSI reached overbought territory ({rsi:.1f}) in {symbol}."
            })
        elif rsi <= self.thresholds["rsi_oversold"]:
            alerts.append({
                "type": "RSI_EXTREME_OVERSOLD",
                "severity": "MEDIUM",
                "symbol": symbol,
                "message": f"RSI reached oversold territory ({rsi:.1f}) in {symbol}."
            })

        if abs(gs_ratio_zscore) >= self.thresholds["gold_silver_ratio_zscore_extreme"]:
            alerts.append({
                "type": "GOLD_SILVER_RATIO_EXTREME",
                "severity": "HIGH",
                "symbol": "GOLD_SILVER_RATIO",
                "message": f"Gold/Silver Ratio Z-score reached extreme level ({gs_ratio_zscore:.2f})."
            })

        if tech_signals.get("breakout_20d"):
            alerts.append({
                "type": "TECHNICAL_BREAKOUT",
                "severity": "HIGH",
                "symbol": symbol,
                "message": f"20-day technical price breakout triggered for {symbol}."
            })

        return alerts
