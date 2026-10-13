"""Alert detection engine for market breakouts, extremes, and regime transitions."""

import os
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class AlertDetector:
    """Detects market breakouts, breakdowns, extreme RSI/volatility, extreme Gold/Silver ratios, and macro regime shifts."""

    @staticmethod
    def detect_alerts(
        date_str: str,
        gold_summary: Dict[str, Any],
        silver_summary: Dict[str, Any],
        ratio_summary: Dict[str, Any],
        macro_summary: Dict[str, Any],
        alert_config: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Evaluates thresholds and triggers structured alert notifications."""

        if alert_config is None:
            alert_config = {
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "gold_silver_ratio_high": 85,
                "gold_silver_ratio_low": 65,
                "daily_move_threshold_pct": 2.0
            }

        alerts = []

        # 1. Gold Daily Move & RSI
        g_move = gold_summary.get("return_1d_pct", 0.0)
        g_rsi = gold_summary.get("rsi_14", 50.0)

        if abs(g_move) >= alert_config.get("daily_move_threshold_pct", 2.0):
            alerts.append({
                "date": date_str,
                "type": "GOLD_LARGE_MOVE",
                "severity": "HIGH",
                "message": f"Gold experienced a significant daily move of {g_move:+.2f}%"
            })

        if g_rsi >= alert_config.get("rsi_overbought", 70):
            alerts.append({
                "date": date_str,
                "type": "GOLD_RSI_OVERBOUGHT",
                "severity": "MEDIUM",
                "message": f"Gold RSI reached overbought condition at {g_rsi:.1f}"
            })
        elif g_rsi <= alert_config.get("rsi_oversold", 30):
            alerts.append({
                "date": date_str,
                "type": "GOLD_RSI_OVERSOLD",
                "severity": "MEDIUM",
                "message": f"Gold RSI reached oversold condition at {g_rsi:.1f}"
            })

        # 2. Silver Daily Move & RSI
        s_move = silver_summary.get("return_1d_pct", 0.0)
        s_rsi = silver_summary.get("rsi_14", 50.0)

        if abs(s_move) >= alert_config.get("daily_move_threshold_pct", 2.0):
            alerts.append({
                "date": date_str,
                "type": "SILVER_LARGE_MOVE",
                "severity": "HIGH",
                "message": f"Silver experienced a significant daily move of {s_move:+.2f}%"
            })

        if s_rsi >= alert_config.get("rsi_overbought", 70):
            alerts.append({
                "date": date_str,
                "type": "SILVER_RSI_OVERBOUGHT",
                "severity": "MEDIUM",
                "message": f"Silver RSI reached overbought condition at {s_rsi:.1f}"
            })
        elif s_rsi <= alert_config.get("rsi_oversold", 30):
            alerts.append({
                "date": date_str,
                "type": "SILVER_RSI_OVERSOLD",
                "severity": "MEDIUM",
                "message": f"Silver RSI reached oversold condition at {s_rsi:.1f}"
            })

        # 3. Gold/Silver Ratio Extreme
        ratio = ratio_summary.get("current_ratio", 0.0)
        if ratio >= alert_config.get("gold_silver_ratio_high", 85):
            alerts.append({
                "date": date_str,
                "type": "RATIO_EXTREME_HIGH",
                "severity": "HIGH",
                "message": f"Gold/Silver ratio reached extreme high level of {ratio:.2f}"
            })
        elif ratio > 0 and ratio <= alert_config.get("gold_silver_ratio_low", 65):
            alerts.append({
                "date": date_str,
                "type": "RATIO_EXTREME_LOW",
                "severity": "HIGH",
                "message": f"Gold/Silver ratio reached extreme low level of {ratio:.2f}"
            })

        return alerts

    @staticmethod
    def save_alerts(date_str: str, alerts: List[Dict[str, Any]], output_dir: str = "data/analysis/alerts") -> str:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{date_str}.json")
        with open(file_path, "w") as f:
            json.dump({"date": date_str, "alert_count": len(alerts), "alerts": alerts}, f, indent=2)
        logger.info(f"Saved alerts to {file_path}")
        return file_path
