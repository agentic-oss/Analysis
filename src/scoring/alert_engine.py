import logging
import pandas as pd

logger = logging.getLogger(__name__)

class AlertEngine:
    """Detects significant market events, breakouts, breakdowns, ratio extremes, and macro regime shifts based on configurable thresholds."""

    def __init__(self, thresholds: dict = None):
        self.thresholds = thresholds or {
            "price_change_1d_pct": 2.0,
            "rsi_oversold": 30.0,
            "rsi_overbought": 70.0,
            "gold_silver_ratio_high": 85.0,
            "gold_silver_ratio_low": 65.0,
            "dxy_change_1d_pct": 1.0,
            "vix_threshold_high": 25.0
        }

    def detect_alerts(
        self,
        date_str: str,
        gold_df: pd.DataFrame,
        silver_df: pd.DataFrame,
        ratio_summary: dict,
        macro_dict: dict
    ) -> dict:
        """
        Scans current market state for significant conditions exceeding configured thresholds.
        Returns dictionary of triggered alerts.
        """
        alerts = []

        # Gold checks
        if not gold_df.empty:
            g_latest = gold_df.iloc[-1]
            g_change = g_latest.get("return_1d", 0.0) or 0.0
            g_rsi = g_latest.get("rsi_14", 50.0) or 50.0

            if abs(g_change * 100) >= self.thresholds["price_change_1d_pct"]:
                alerts.append({
                    "symbol": "GOLD",
                    "alert_type": "LARGE_PRICE_MOVE",
                    "severity": "HIGH",
                    "message": f"Gold 1-day price move of {g_change*100:+.2f}% exceeds threshold ({self.thresholds['price_change_1d_pct']}%)"
                })
            if g_latest.get("breakout_20d", False):
                alerts.append({
                    "symbol": "GOLD",
                    "alert_type": "BREAKOUT_20D",
                    "severity": "MEDIUM",
                    "message": "Gold technical 20-day price breakout detected."
                })
            if g_latest.get("breakdown_20d", False):
                alerts.append({
                    "symbol": "GOLD",
                    "alert_type": "BREAKDOWN_20D",
                    "severity": "MEDIUM",
                    "message": "Gold technical 20-day price breakdown detected."
                })
            if g_rsi >= self.thresholds["rsi_overbought"]:
                alerts.append({
                    "symbol": "GOLD",
                    "alert_type": "EXTREME_RSI_OVERBOUGHT",
                    "severity": "MEDIUM",
                    "message": f"Gold RSI is overbought at {g_rsi:.1f} (>= {self.thresholds['rsi_overbought']})"
                })
            elif g_rsi <= self.thresholds["rsi_oversold"]:
                alerts.append({
                    "symbol": "GOLD",
                    "alert_type": "EXTREME_RSI_OVERSOLD",
                    "severity": "MEDIUM",
                    "message": f"Gold RSI is oversold at {g_rsi:.1f} (<= {self.thresholds['rsi_oversold']})"
                })

        # Silver checks
        if not silver_df.empty:
            s_latest = silver_df.iloc[-1]
            s_change = s_latest.get("return_1d", 0.0) or 0.0
            s_rsi = s_latest.get("rsi_14", 50.0) or 50.0

            if abs(s_change * 100) >= self.thresholds["price_change_1d_pct"]:
                alerts.append({
                    "symbol": "SILVER",
                    "alert_type": "LARGE_PRICE_MOVE",
                    "severity": "HIGH",
                    "message": f"Silver 1-day price move of {s_change*100:+.2f}% exceeds threshold ({self.thresholds['price_change_1d_pct']}%)"
                })
            if s_latest.get("breakout_20d", False):
                alerts.append({
                    "symbol": "SILVER",
                    "alert_type": "BREAKOUT_20D",
                    "severity": "MEDIUM",
                    "message": "Silver technical 20-day price breakout detected."
                })
            if s_latest.get("breakdown_20d", False):
                alerts.append({
                    "symbol": "SILVER",
                    "alert_type": "BREAKDOWN_20D",
                    "severity": "MEDIUM",
                    "message": "Silver technical 20-day price breakdown detected."
                })

        # Ratio checks
        if ratio_summary:
            r_val = ratio_summary.get("current_ratio", 0.0)
            z_val = ratio_summary.get("ratio_zscore", 0.0)
            if r_val >= self.thresholds["gold_silver_ratio_high"] or abs(z_val) >= 2.0:
                alerts.append({
                    "symbol": "RATIO",
                    "alert_type": "GOLD_SILVER_RATIO_EXTREME",
                    "severity": "HIGH",
                    "message": f"Gold/Silver ratio extreme level detected: {r_val:.2f} (Z-Score: {z_val:.2f})"
                })

        # Macro checks
        vix_lvl = macro_dict.get("components", {}).get("vix_level", 0.0)
        if vix_lvl >= self.thresholds["vix_threshold_high"]:
            alerts.append({
                "symbol": "MACRO",
                "alert_type": "HIGH_VOLATILITY_VIX",
                "severity": "HIGH",
                "message": f"VIX elevated at {vix_lvl:.1f} (>= {self.thresholds['vix_threshold_high']})"
            })

        return {
            "date": date_str,
            "alerts_count": len(alerts),
            "alerts": alerts
        }
