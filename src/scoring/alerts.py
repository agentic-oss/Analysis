import logging
import pandas as pd
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class AlertEngine:
    """Detects major technical, macro, and ratio thresholds and generates alert events."""

    def __init__(self, config_alerts: Dict[str, Any]):
        self.config = config_alerts

    def detect_alerts(
        self,
        date_str: str,
        gold_df: pd.DataFrame,
        silver_df: pd.DataFrame,
        ratio_val: float,
        dxy_return: float,
        real_yield_change: float
    ) -> Dict[str, Any]:
        alerts = []

        # Gold RSI Extreme
        if not gold_df.empty and 'rsi_14' in gold_df.columns:
            rsi = gold_df['rsi_14'].iloc[-1]
            if rsi >= self.config.get("rsi_overbought", 70):
                alerts.append({"type": "GOLD_RSI_OVERBOUGHT", "severity": "HIGH", "message": f"Gold RSI reached overbought level: {rsi:.1f}"})
            elif rsi <= self.config.get("rsi_oversold", 30):
                alerts.append({"type": "GOLD_RSI_OVERSOLD", "severity": "HIGH", "message": f"Gold RSI reached oversold level: {rsi:.1f}"})

        # Ratio Extreme
        if ratio_val >= self.config.get("gold_silver_ratio_extreme_high", 90):
            alerts.append({"type": "RATIO_EXTREME_HIGH", "severity": "MEDIUM", "message": f"Gold/Silver Ratio reached extreme high level: {ratio_val:.2f}"})
        elif ratio_val <= self.config.get("gold_silver_ratio_extreme_low", 65):
            alerts.append({"type": "RATIO_EXTREME_LOW", "severity": "MEDIUM", "message": f"Gold/Silver Ratio reached extreme low level: {ratio_val:.2f}"})

        # Yield shock
        if abs(real_yield_change) > 0.15:
            alerts.append({"type": "REAL_YIELD_SHOCK", "severity": "HIGH", "message": f"Significant 20d real yield shift: {real_yield_change:+.2f}"})

        return {
            "date": date_str,
            "alerts_count": len(alerts),
            "alerts": alerts
        }
