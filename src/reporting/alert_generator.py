import os
import json
import logging
import pandas as pd
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class AlertGenerator:
    """Detects significant threshold breaches and triggers alerts."""

    def __init__(self, output_dir: str = "data/analysis/alerts"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def check_alerts(self, master_df: pd.DataFrame, current_idx: int) -> List[Dict[str, Any]]:
        if current_idx < 1 or current_idx >= len(master_df):
            return []

        curr = master_df.iloc[current_idx]
        prev = master_df.iloc[current_idx - 1]
        date_str = str(curr["date"])

        alerts = []

        # 1. Price Moves (> 2.5%)
        for sym in ["GOLD", "SILVER"]:
            col = f"{sym}_close"
            if col in curr and col in prev and not pd.isna(curr[col]) and not pd.isna(prev[col]):
                pct_chg = (curr[col] - prev[col]) / prev[col]
                if abs(pct_chg) >= 0.025:
                    alert_type = f"{sym}_BREAKOUT" if pct_chg > 0 else f"{sym}_BREAKDOWN"
                    alerts.append({
                        "date": date_str,
                        "type": alert_type,
                        "severity": "HIGH",
                        "message": f"{sym} single-day move of {pct_chg*100:.2f}% (Price: {curr[col]:.2f})"
                    })

        # 2. RSI Extremes
        for sym in ["gold_", "silver_"]:
            rsi_col = f"{sym}rsi_14"
            if rsi_col in curr and not pd.isna(curr[rsi_col]):
                rsi_val = curr[rsi_col]
                if rsi_val >= 70:
                    alerts.append({
                        "date": date_str,
                        "type": "EXTREME_RSI_OVERBOUGHT",
                        "severity": "MEDIUM",
                        "message": f"{sym.upper()} RSI reached overbought level: {rsi_val:.1f}"
                    })
                elif rsi_val <= 30:
                    alerts.append({
                        "date": date_str,
                        "type": "EXTREME_RSI_OVERSOLD",
                        "severity": "MEDIUM",
                        "message": f"{sym.upper()} RSI reached oversold level: {rsi_val:.1f}"
                    })

        # 3. Gold/Silver Ratio Z-score extreme
        if "ratio_zscore_252d" in curr and not pd.isna(curr["ratio_zscore_252d"]):
            z = curr["ratio_zscore_252d"]
            if abs(z) >= 2.0:
                alerts.append({
                    "date": date_str,
                    "type": "RATIO_EXTREME_ZSCORE",
                    "severity": "HIGH",
                    "message": f"Gold/Silver ratio 252d Z-Score extreme: {z:.2f}"
                })

        # Save alerts to JSON
        file_path = os.path.join(self.output_dir, f"{date_str}.json")
        with open(file_path, "w") as f:
            json.dump({"date": date_str, "alerts_count": len(alerts), "alerts": alerts}, f, indent=2)

        return alerts
