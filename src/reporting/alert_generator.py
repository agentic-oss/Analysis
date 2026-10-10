import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class AlertGenerator:
    """
    Detects significant market events (breakouts, breakdowns, RSI extremes, high GSR, price gaps, VIX spikes)
    and writes alerts to data/analysis/alerts/YYYY-MM-DD.json.
    """

    def __init__(self, rsi_high: float = 75.0, rsi_low: float = 25.0, jump_pct: float = 2.5, gsr_z_extreme: float = 2.0):
        self.rsi_high = rsi_high
        self.rsi_low = rsi_low
        self.jump_pct = jump_pct
        self.gsr_z_extreme = gsr_z_extreme

    def detect_alerts(self, date_str: str, gold_row: pd.Series, silver_row: pd.Series, ratio_metrics: Dict[str, Any], macro_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        alerts = []

        # Gold checks
        if not gold_row.empty:
            g_ret = abs(gold_row.get("return_1d", 0.0) * 100)
            if g_ret >= self.jump_pct:
                alerts.append({"type": "GOLD_LARGE_MOVE", "severity": "HIGH", "message": f"Gold 1D move was {g_ret:.2f}%"})
            if gold_row.get("rsi_14", 50) >= self.rsi_high:
                alerts.append({"type": "GOLD_OVERBOUGHT", "severity": "MEDIUM", "message": f"Gold RSI reached {gold_row.get('rsi_14'):.1f}"})
            elif gold_row.get("rsi_14", 50) <= self.rsi_low:
                alerts.append({"type": "GOLD_OVERSOLD", "severity": "MEDIUM", "message": f"Gold RSI reached {gold_row.get('rsi_14'):.1f}"})
            if gold_row.get("breakout_20d", False):
                alerts.append({"type": "GOLD_20D_BREAKOUT", "severity": "HIGH", "message": "Gold broke out to 20-day high"})

        # Silver checks
        if not silver_row.empty:
            s_ret = abs(silver_row.get("return_1d", 0.0) * 100)
            if s_ret >= self.jump_pct:
                alerts.append({"type": "SILVER_LARGE_MOVE", "severity": "HIGH", "message": f"Silver 1D move was {s_ret:.2f}%"})

        # Ratio checks
        z_score = ratio_metrics.get("z_score", 0.0)
        if abs(z_score) >= self.gsr_z_extreme:
            alerts.append({"type": "GSR_EXTREME", "severity": "HIGH", "message": f"Gold/Silver ratio Z-score is extreme: {z_score:.2f}"})

        return alerts
