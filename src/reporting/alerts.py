"""
Alert Detection Engine.
Triggers alerts for key market anomalies and threshold breaches.
"""
from typing import Dict, List, Any


class AlertEngine:
    """
    Detects significant market events (breakouts, technical extremes, ratio z-score extremes, macro regime shifts).
    """

    @staticmethod
    def detect_alerts(
        date_str: str,
        gold_analysis: Dict[str, Any],
        silver_analysis: Dict[str, Any],
        ratio_analysis: Dict[str, Any],
        macro_regime: Dict[str, Any],
    ) -> Dict[str, Any]:

        alerts = []

        # Gold RSI Extremes
        g_rsi = gold_analysis.get("rsi_14", 50.0)
        if g_rsi >= 70.0:
            alerts.append({"type": "GOLD_RSI_OVERBOUGHT", "severity": "MEDIUM", "message": f"Gold RSI is overbought at {g_rsi:.1f}"})
        elif g_rsi <= 30.0:
            alerts.append({"type": "GOLD_RSI_OVERSOLD", "severity": "HIGH", "message": f"Gold RSI is oversold at {g_rsi:.1f}"})

        # Silver RSI Extremes
        s_rsi = silver_analysis.get("rsi_14", 50.0)
        if s_rsi >= 70.0:
            alerts.append({"type": "SILVER_RSI_OVERBOUGHT", "severity": "MEDIUM", "message": f"Silver RSI is overbought at {s_rsi:.1f}"})
        elif s_rsi <= 30.0:
            alerts.append({"type": "SILVER_RSI_OVERSOLD", "severity": "HIGH", "message": f"Silver RSI is oversold at {s_rsi:.1f}"})

        # Ratio Extremes
        ratio_z = ratio_analysis.get("zscore", 0.0)
        if abs(ratio_z) >= 2.0:
            alerts.append({
                "type": "RATIO_EXTREME",
                "severity": "HIGH",
                "message": f"Gold/Silver Ratio Z-score reached extreme level: {ratio_z:+.2f}",
            })

        # Golden Cross / Death Cross
        if gold_analysis.get("golden_cross", False):
            alerts.append({"type": "GOLD_GOLDEN_CROSS", "severity": "HIGH", "message": "Gold 50 DMA crossed above 200 DMA (Golden Cross)"})
        if gold_analysis.get("death_cross", False):
            alerts.append({"type": "GOLD_DEATH_CROSS", "severity": "HIGH", "message": "Gold 50 DMA crossed below 200 DMA (Death Cross)"})

        return {
            "date": date_str,
            "total_alerts": len(alerts),
            "alerts": alerts,
        }
