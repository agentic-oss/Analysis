"""
Transparent Multi-Factor Sub-Scoring Engine (0-100) and Alert Generation Engine.
Stores all individual components for total explainability.
"""
import logging
from typing import Dict, List, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FactorScorer:
    """
    Computes transparent multi-factor sub-scores (0 to 100) and composite weighted score:
      - Technical Score (0-100)
      - Momentum Score (0-100)
      - Volatility Score (0-100)
      - Macro Score (0-100)
      - Relative Value Score (0-100)
      - Historical Pattern Score (0-100)
    """

    @staticmethod
    def calculate_scores(
        tech_df: pd.DataFrame,
        macro_regime_info: Dict[str, Any],
        relative_value_df: pd.DataFrame,
        analogue_info: Dict[str, Any],
        weights: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Calculates sub-scores and weighted composite research score.
        """
        default_weights = {
            "technical": 0.20,
            "macro": 0.20,
            "momentum": 0.15,
            "volatility": 0.15,
            "relative_value": 0.15,
            "historical_pattern": 0.15
        }
        w = weights or default_weights

        if tech_df.empty:
            return {
                "composite_score": 50.0,
                "sub_scores": {
                    "technical": 50.0,
                    "macro": 50.0,
                    "momentum": 50.0,
                    "volatility": 50.0,
                    "relative_value": 50.0,
                    "historical_pattern": 50.0
                }
            }

        last = tech_df.iloc[-1]
        close = float(last.get("close", 0))

        # 1. Technical Score (Moving Average alignment & distance)
        tech_points = 50.0
        if "sma_20" in last and close > float(last["sma_20"]):
            tech_points += 15.0
        if "sma_50" in last and close > float(last["sma_50"]):
            tech_points += 15.0
        if "sma_200" in last and close > float(last["sma_200"]):
            tech_points += 20.0
        tech_score = max(0.0, min(100.0, tech_points))

        # 2. Momentum Score (RSI & MACD)
        rsi = float(last.get("rsi_14", 50.0))
        macd_hist = float(last.get("macd_hist", 0.0))
        mom_points = 50.0 + (rsi - 50.0) * 0.5
        if macd_hist > 0:
            mom_points += 15.0
        else:
            mom_points -= 15.0
        momentum_score = max(0.0, min(100.0, mom_points))

        # 3. Volatility Score (Lower/Normal volatility gives higher trend stability score)
        vol_20d = float(last.get("volatility_20d", 0.15))
        if vol_20d < 0.10:
            vol_score = 80.0
        elif vol_20d < 0.20:
            vol_score = 65.0
        elif vol_20d < 0.30:
            vol_score = 45.0
        else:
            vol_score = 25.0

        # 4. Macro Score
        regime = macro_regime_info.get("macro_regime", "Neutral")
        macro_score_map = {
            "Inflationary": 85.0,
            "Stagflationary": 90.0,
            "Risk-Off": 80.0,
            "Disinflationary": 60.0,
            "Easing": 75.0,
            "Tightening": 35.0,
            "Risk-On": 40.0,
            "Neutral": 50.0
        }
        macro_score = macro_score_map.get(regime, 50.0)

        # 5. Relative Value Score
        rv_score = 50.0
        if not relative_value_df.empty and "ratio_zscore_60d" in relative_value_df.columns:
            z = float(relative_value_df["ratio_zscore_60d"].iloc[-1])
            if abs(z) > 1.5:
                rv_score = 75.0
            if abs(z) > 2.0:
                rv_score = 90.0
        relative_value_score = rv_score

        # 6. Historical Pattern Score (Positive return probability over 10d)
        f_stats = analogue_info.get("forward_statistics", {}).get("horizon_10d", {})
        pos_prob = f_stats.get("positive_prob_pct", 50.0)
        pattern_score = max(0.0, min(100.0, pos_prob))

        sub_scores = {
            "technical": round(tech_score, 2),
            "macro": round(macro_score, 2),
            "momentum": round(momentum_score, 2),
            "volatility": round(vol_score, 2),
            "relative_value": round(relative_value_score, 2),
            "historical_pattern": round(pattern_score, 2)
        }

        composite = (
            sub_scores["technical"] * w["technical"] +
            sub_scores["macro"] * w["macro"] +
            sub_scores["momentum"] * w["momentum"] +
            sub_scores["volatility"] * w["volatility"] +
            sub_scores["relative_value"] * w["relative_value"] +
            sub_scores["historical_pattern"] * w["historical_pattern"]
        )

        return {
            "composite_score": round(composite, 2),
            "sub_scores": sub_scores,
            "weights": w
        }


class AlertDetector:
    """
    Alert Detection System for identifying market anomalies, breakouts, RSI extremes,
    and relative value extremes.
    """

    @staticmethod
    def detect_alerts(
        date_str: str,
        symbol: str,
        tech_df: pd.DataFrame,
        relative_value_df: pd.DataFrame = None
    ) -> List[Dict[str, Any]]:
        alerts = []
        if tech_df.empty:
            return alerts

        last = tech_df.iloc[-1]
        rsi = float(last.get("rsi_14", 50.0))
        ret_1d = float(last.get("return_1d", 0.0)) * 100.0

        if rsi > 70:
            alerts.append({
                "date": date_str,
                "symbol": symbol,
                "alert_type": "extreme_rsi_overbought",
                "severity": "medium",
                "message": f"{symbol} RSI (14) is overbought at {rsi:.1f}"
            })
        elif rsi < 30:
            alerts.append({
                "date": date_str,
                "symbol": symbol,
                "alert_type": "extreme_rsi_oversold",
                "severity": "medium",
                "message": f"{symbol} RSI (14) is oversold at {rsi:.1f}"
            })

        if bool(last.get("breakout_20d", False)):
            alerts.append({
                "date": date_str,
                "symbol": symbol,
                "alert_type": "breakout_20d",
                "severity": "high",
                "message": f"{symbol} triggered a 20-day high breakout"
            })
        elif bool(last.get("breakdown_20d", False)):
            alerts.append({
                "date": date_str,
                "symbol": symbol,
                "alert_type": "breakdown_20d",
                "severity": "high",
                "message": f"{symbol} triggered a 20-day low breakdown"
            })

        if abs(ret_1d) >= 3.0:
            alerts.append({
                "date": date_str,
                "symbol": symbol,
                "alert_type": "large_price_move",
                "severity": "high",
                "message": f"{symbol} experienced a large 1-day move of {ret_1d:+.2f}%"
            })

        if relative_value_df is not None and not relative_value_df.empty:
            last_rv = relative_value_df.iloc[-1]
            ratio = float(last_rv.get("gold_silver_ratio", 0))
            if ratio >= 85.0:
                alerts.append({
                    "date": date_str,
                    "symbol": symbol,
                    "alert_type": "gold_silver_ratio_extreme_high",
                    "severity": "medium",
                    "message": f"Gold/Silver Ratio is extremely high at {ratio:.2f}"
                })
            elif ratio > 0 and ratio <= 65.0:
                alerts.append({
                    "date": date_str,
                    "symbol": symbol,
                    "alert_type": "gold_silver_ratio_extreme_low",
                    "severity": "medium",
                    "message": f"Gold/Silver Ratio is extremely low at {ratio:.2f}"
                })

        return alerts
