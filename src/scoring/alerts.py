import os
import json
from typing import Dict, Any, List

class AlertDetector:
    """Detects major market events and writes data/analysis/alerts/YYYY-MM-DD.json."""

    def __init__(self, alerts_dir: str = "data/analysis/alerts"):
        self.alerts_dir = alerts_dir
        os.makedirs(self.alerts_dir, exist_ok=True)

    def detect_alerts(
        self,
        date_str: str,
        gold_data: Dict[str, Any],
        silver_data: Dict[str, Any],
        gsr_data: Dict[str, Any],
        macro_data: Dict[str, Any]
    ) -> Dict[str, Any]:

        alerts = []

        # Gold RSI Extreme
        g_rsi = gold_data.get("rsi_14", 50.0)
        if g_rsi >= 70:
            alerts.append({"type": "GOLD_RSI_OVERBOUGHT", "severity": "MEDIUM", "message": f"Gold RSI is overbought at {g_rsi:.1f}"})
        elif g_rsi <= 30:
            alerts.append({"type": "GOLD_RSI_OVERSOLD", "severity": "HIGH", "message": f"Gold RSI is oversold at {g_rsi:.1f}"})

        # Silver RSI Extreme
        s_rsi = silver_data.get("rsi_14", 50.0)
        if s_rsi >= 70:
            alerts.append({"type": "SILVER_RSI_OVERBOUGHT", "severity": "MEDIUM", "message": f"Silver RSI is overbought at {s_rsi:.1f}"})
        elif s_rsi <= 30:
            alerts.append({"type": "SILVER_RSI_OVERSOLD", "severity": "HIGH", "message": f"Silver RSI is oversold at {s_rsi:.1f}"})

        # GSR Extreme
        gsr_z = gsr_data.get("z_score", 0.0)
        if abs(gsr_z) >= 2.0:
            alerts.append({"type": "GOLD_SILVER_RATIO_EXTREME", "severity": "HIGH", "message": f"Gold/Silver ratio reached extreme Z-score of {gsr_z:.2f}"})

        # Macro Regime Change
        regime = macro_data.get("regime", "Neutral")
        if regime in ["Stagflationary", "Risk-Off"]:
            alerts.append({"type": "MACRO_HIGH_RISK_REGIME", "severity": "HIGH", "message": f"Macro regime shifted to high-risk state: {regime}"})

        result = {
            "date": date_str,
            "total_alerts": len(alerts),
            "alerts": alerts
        }

        out_path = os.path.join(self.alerts_dir, f"{date_str}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        return result
