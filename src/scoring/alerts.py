"""
Alert detection engine identifying significant price, technical, and macro events.
Outputs alerts to data/analysis/alerts/YYYY-MM-DD.json.
"""

import os
import json
import logging
import pandas as pd
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class AlertEngine:
    """Detects market breakouts, extreme RSI, ratio extremes, and macro transitions."""

    def __init__(self, thresholds: Dict[str, float] = None):
        self.thresholds = thresholds or {
            "rsi_overbought": 70.0,
            "rsi_oversold": 30.0,
            "dxy_daily_change_pct": 0.01,
            "ratio_zscore_extreme": 2.0,
        }

    def detect_alerts(self, date_str: str, feature_row: pd.Series, output_dir: str = "data/analysis/alerts") -> Dict[str, Any]:
        """Scans feature_row for alert conditions and saves alert JSON."""
        alerts = []

        # 1. Extreme RSI
        rsi = feature_row.get("rsi_14")
        if pd.notna(rsi):
            if rsi >= self.thresholds["rsi_overbought"]:
                alerts.append({
                    "severity": "MEDIUM",
                    "type": "EXTREME_RSI_OVERBOUGHT",
                    "message": f"Gold RSI 14 reached overbought level: {rsi:.1f}",
                })
            elif rsi <= self.thresholds["rsi_oversold"]:
                alerts.append({
                    "severity": "MEDIUM",
                    "type": "EXTREME_RSI_OVERSOLD",
                    "message": f"Gold RSI 14 reached oversold level: {rsi:.1f}",
                })

        # 2. Gold/Silver ratio extreme
        ratio_z = feature_row.get("gold_silver_ratio_zscore")
        if pd.notna(ratio_z):
            if abs(ratio_z) >= self.thresholds["ratio_zscore_extreme"]:
                alerts.append({
                    "severity": "HIGH",
                    "type": "GOLD_SILVER_RATIO_EXTREME",
                    "message": f"Gold/Silver ratio Z-score extreme: {ratio_z:.2f}",
                })

        # 3. Golden Cross / Death Cross
        if feature_row.get("golden_cross", False):
            alerts.append({
                "severity": "HIGH",
                "type": "GOLDEN_CROSS",
                "message": "50-day SMA crossed above 200-day SMA (Golden Cross).",
            })
        if feature_row.get("death_cross", False):
            alerts.append({
                "severity": "HIGH",
                "type": "DEATH_CROSS",
                "message": "50-day SMA crossed below 200-day SMA (Death Cross).",
            })

        result = {
            "date": date_str,
            "alert_count": len(alerts),
            "alerts": alerts,
        }

        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{date_str}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        return result
