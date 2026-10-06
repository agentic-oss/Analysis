import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class AlertDetector:
    """Detects technical breakouts, breakdowns, RSI extremes, volatility spikes, and ratio extremes based on configurable thresholds."""

    def __init__(self, thresholds: Dict[str, float] = None):
        self.thresholds = thresholds or {
            "rsi_overbought": 70.0,
            "rsi_oversold": 30.0,
            "ratio_zscore_extreme": 2.0,
            "price_change_large_pct": 2.5
        }

    def detect_alerts(self, date_str: str, tech_metrics: Dict[str, Any], ratio_info: Dict[str, Any], instrument: str = "GOLD") -> Dict[str, Any]:
        alerts = []

        rsi = tech_metrics.get("rsi_14", 50.0)
        if rsi >= self.thresholds.get("rsi_overbought", 70.0):
            alerts.append({"type": "RSI_OVERBOUGHT", "level": "WARNING", "message": f"{instrument} RSI is overbought at {rsi:.1f}"})
        elif rsi <= self.thresholds.get("rsi_oversold", 30.0):
            alerts.append({"type": "RSI_OVERSOLD", "level": "WARNING", "message": f"{instrument} RSI is oversold at {rsi:.1f}"})

        ret_1d = abs(tech_metrics.get("return_1d", 0.0) or 0.0) * 100.0
        if ret_1d >= self.thresholds.get("price_change_large_pct", 2.5):
            alerts.append({"type": "LARGE_PRICE_MOVE", "level": "INFO", "message": f"{instrument} experienced large 1D move of {ret_1d:.2f}%"})

        zscore = abs(ratio_info.get("ratio_zscore", 0.0))
        if zscore >= self.thresholds.get("ratio_zscore_extreme", 2.0):
            alerts.append({"type": "GOLD_SILVER_RATIO_EXTREME", "level": "WARNING", "message": f"Gold/Silver ratio z-score extreme at {zscore:.2f}"})

        return {
            "date": date_str,
            "instrument": instrument,
            "alerts_triggered": len(alerts),
            "alerts": alerts
        }
