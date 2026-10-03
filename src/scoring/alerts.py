import json
import logging
from typing import Dict, Any, List
import pandas as pd

logger = logging.getLogger(__name__)


class AlertDetector:
    """Detects significant market alerts and writes data/analysis/alerts/YYYY-MM-DD.json."""

    @staticmethod
    def detect_alerts(
        gold_df: pd.DataFrame,
        silver_df: pd.DataFrame,
        macro_summary: Dict[str, Any],
        ratio_summary: Dict[str, Any],
        date_str: str,
    ) -> Dict[str, Any]:
        alerts = []

        # Gold checks
        if gold_df is not None and not gold_df.empty:
            g_latest = gold_df.iloc[-1]
            if g_latest.get("signal_breakout_20d"):
                alerts.append({"type": "gold_breakout", "level": "high", "message": "Gold broke out to 20-day high."})
            if g_latest.get("signal_breakdown_20d"):
                alerts.append({"type": "gold_breakdown", "level": "high", "message": "Gold broke down to 20-day low."})
            rsi = g_latest.get("rsi_14", 50.0)
            if rsi > 70:
                alerts.append({"type": "gold_rsi_overbought", "level": "medium", "message": f"Gold RSI is overbought ({rsi:.1f})."})
            elif rsi < 30:
                alerts.append({"type": "gold_rsi_oversold", "level": "medium", "message": f"Gold RSI is oversold ({rsi:.1f})."})
            gap = abs(g_latest.get("gap", 0.0)) * 100
            if gap > 1.0:
                alerts.append({"type": "gold_price_gap", "level": "medium", "message": f"Gold experienced a {gap:.2f}% price gap at open."})

        # Silver checks
        if silver_df is not None and not silver_df.empty:
            s_latest = silver_df.iloc[-1]
            if s_latest.get("signal_breakout_20d"):
                alerts.append({"type": "silver_breakout", "level": "high", "message": "Silver broke out to 20-day high."})
            if s_latest.get("signal_breakdown_20d"):
                alerts.append({"type": "silver_breakdown", "level": "high", "message": "Silver broke down to 20-day low."})
            rsi = s_latest.get("rsi_14", 50.0)
            if rsi > 70:
                alerts.append({"type": "silver_rsi_overbought", "level": "medium", "message": f"Silver RSI is overbought ({rsi:.1f})."})
            elif rsi < 30:
                alerts.append({"type": "silver_rsi_oversold", "level": "medium", "message": f"Silver RSI is oversold ({rsi:.1f})."})

        # Gold/Silver Ratio extremes
        if ratio_summary and "zscore" in ratio_summary:
            zs = ratio_summary["zscore"]
            if abs(zs) > 2.0:
                alerts.append({
                    "type": "gsr_extreme",
                    "level": "high",
                    "message": f"Gold/Silver ratio reaches extreme level (Z-score: {zs:.2f}, Ratio: {ratio_summary['current_ratio']}).",
                })

        # Macro checks
        if macro_summary:
            dxy_pct = abs(macro_summary.get("dxy", {}).get("pct_change", 0.0))
            if dxy_pct > 1.0:
                alerts.append({"type": "dxy_extreme_move", "level": "medium", "message": f"US Dollar Index moved {dxy_pct:.2f}%."})
            yield_chg = abs(macro_summary.get("us10y", {}).get("abs_change", 0.0))
            if yield_chg > 0.10:
                alerts.append({"type": "us10y_extreme_move", "level": "medium", "message": f"US 10Y Yield moved {yield_chg:.2f} percentage points."})

        return {
            "date": date_str,
            "alert_count": len(alerts),
            "alerts": alerts,
        }
