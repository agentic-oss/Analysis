import json
import os
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class AlertEngine:
    """Alert detection engine evaluating market anomalies, breakouts, ratio extremes, and macro shocks."""

    def __init__(
        self,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        daily_price_jump_pct: float = 3.0,
        dxy_daily_change_pct: float = 1.0,
        ratio_zscore_extreme: float = 2.0
    ):
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.daily_price_jump_pct = daily_price_jump_pct
        self.dxy_daily_change_pct = dxy_daily_change_pct
        self.ratio_zscore_extreme = ratio_zscore_extreme

    def detect_alerts(
        self,
        date_str: str,
        metals_data: Dict[str, Any],
        ratio_data: Dict[str, Any] = None,
        macro_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        alerts = []

        # 1. Metals Technical & Price Alerts
        for sym in ["GOLD", "SILVER"]:
            m = metals_data.get(sym.lower())
            if not m:
                continue

            # Price Jump Alert
            daily_move = m.get("daily_move_pct", 0.0)
            if abs(daily_move) >= self.daily_price_jump_pct:
                alerts.append({
                    "symbol": sym,
                    "type": "LARGE_PRICE_MOVE",
                    "severity": "HIGH",
                    "message": f"{sym} executed a large daily price move of {daily_move:.2f}%."
                })

            # RSI Alerts
            rsi = m.get("technical_indicators", {}).get("rsi_14", 50.0)
            if rsi >= self.rsi_overbought:
                alerts.append({
                    "symbol": sym,
                    "type": "RSI_OVERBOUGHT",
                    "severity": "MEDIUM",
                    "message": f"{sym} RSI reached overbought level at {rsi:.1f}."
                })
            elif rsi <= self.rsi_oversold:
                alerts.append({
                    "symbol": sym,
                    "type": "RSI_OVERSOLD",
                    "severity": "MEDIUM",
                    "message": f"{sym} RSI reached oversold level at {rsi:.1f}."
                })

            # Volatility Alert
            vol = m.get("volatility_20d_ann", 15.0)
            if vol > 35.0:
                alerts.append({
                    "symbol": sym,
                    "type": "EXTREME_VOLATILITY",
                    "severity": "HIGH",
                    "message": f"{sym} annualized 20-day volatility spiked to {vol:.1f}%."
                })

        # 2. Ratio Extremes Alert
        if ratio_data:
            zscore = ratio_data.get("latest_zscore", 0.0)
            if abs(zscore) >= self.ratio_zscore_extreme:
                alerts.append({
                    "symbol": "GOLD_SILVER_RATIO",
                    "type": "RATIO_EXTREME_ZSCORE",
                    "severity": "HIGH",
                    "message": f"Gold/Silver ratio reached an extreme 252d Z-Score of {zscore:.2f}."
                })

        # Summary
        result = {
            "date": date_str,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "alert_count": len(alerts),
            "alerts": alerts
        }

        return result
