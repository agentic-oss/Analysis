import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class TransparentScoringSystem:
    """Calculates transparent sub-scores (0-100) and weighted composite score."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "technical": 0.20,
            "macro": 0.20,
            "momentum": 0.15,
            "volatility": 0.15,
            "relative_value": 0.15,
            "historical_pattern": 0.15
        }

    def compute_scores(
        self,
        rsi_14: float = 50.0,
        dist_sma_50: float = 0.0,
        macd_hist: float = 0.0,
        macro_regime: str = "Neutral",
        volatility_20d: float = 0.15,
        gs_ratio_zscore: float = 0.0,
        analogue_pos_prob: float = 0.50
    ) -> Dict[str, Any]:
        # Technical score (0-100, 50 neutral)
        tech_score = 50.0 + (dist_sma_50 * 500.0)
        tech_score = float(np.clip(tech_score, 0, 100))

        # Momentum score
        mom_score = 50.0 + (rsi_14 - 50.0) + (macd_hist * 10.0)
        mom_score = float(np.clip(mom_score, 0, 100))

        # Macro score
        macro_map = {
            "Inflationary": 75.0,
            "Stagflationary": 85.0,
            "Easing": 70.0,
            "Risk-Off": 65.0,
            "Disinflationary": 40.0,
            "Tightening": 30.0,
            "Deflationary": 25.0,
            "Neutral": 50.0,
            "Risk-On": 45.0
        }
        macro_score = float(macro_map.get(macro_regime, 50.0))

        # Volatility score (Lower vol -> higher stability score)
        vol_score = 100.0 - (volatility_20d * 300.0)
        vol_score = float(np.clip(vol_score, 0, 100))

        # Relative value score
        rv_score = 50.0 - (gs_ratio_zscore * 15.0)
        rv_score = float(np.clip(rv_score, 0, 100))

        # Pattern score
        pattern_score = float(analogue_pos_prob * 100.0)

        composite = (
            tech_score * self.weights["technical"] +
            macro_score * self.weights["macro"] +
            mom_score * self.weights["momentum"] +
            vol_score * self.weights["volatility"] +
            rv_score * self.weights["relative_value"] +
            pattern_score * self.weights["historical_pattern"]
        )

        return {
            "technical_score": tech_score,
            "macro_score": macro_score,
            "momentum_score": mom_score,
            "volatility_score": vol_score,
            "relative_value_score": rv_score,
            "historical_pattern_score": pattern_score,
            "composite_score": float(composite)
        }


class AlertEngine:
    """Detects market anomalies and saves alerts to data/analysis/alerts/YYYY-MM-DD.json."""

    @staticmethod
    def detect_alerts(
        date_str: str,
        gold_df: pd.DataFrame,
        silver_df: pd.DataFrame,
        macro_dfs: Dict[str, pd.DataFrame],
        output_dir: str = "data/analysis/alerts"
    ) -> List[Dict[str, Any]]:
        alerts = []

        if not gold_df.empty:
            g_close = gold_df["close"].iloc[-1]
            g_ret = gold_df["close"].pct_change().iloc[-1] if len(gold_df) > 1 else 0.0
            if abs(g_ret) > 0.025:
                alerts.append({
                    "date": date_str,
                    "type": "GOLD_LARGE_MOVE",
                    "severity": "HIGH",
                    "message": f"Gold moved {g_ret*100:.2f}% on {date_str}"
                })

        if not silver_df.empty:
            s_ret = silver_df["close"].pct_change().iloc[-1] if len(silver_df) > 1 else 0.0
            if abs(s_ret) > 0.035:
                alerts.append({
                    "date": date_str,
                    "type": "SILVER_LARGE_MOVE",
                    "severity": "HIGH",
                    "message": f"Silver moved {s_ret*100:.2f}% on {date_str}"
                })

        if not gold_df.empty and not silver_df.empty:
            ratio = gold_df["close"].iloc[-1] / silver_df["close"].iloc[-1]
            if ratio > 90.0 or ratio < 65.0:
                alerts.append({
                    "date": date_str,
                    "type": "GOLD_SILVER_RATIO_EXTREME",
                    "severity": "MEDIUM",
                    "message": f"Gold/Silver ratio reached extreme level of {ratio:.2f}"
                })

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{date_str}.json")
        with open(out_path, "w") as f:
            json.dump({"date": date_str, "alerts_count": len(alerts), "alerts": alerts}, f, indent=2)

        return alerts


class ForecastPerformanceTracker:
    """Tracks previous predictions against actual outcomes and monitors performance drift."""

    def __init__(self, storage_path: str = "data/models/model-performance.json"):
        self.storage_path = storage_path

    def update_performance(
        self,
        predictions_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluates historical predictions against realized price returns."""
        if not predictions_history:
            return {}

        df = pd.DataFrame(predictions_history)
        if "actual_return" not in df.columns or df["actual_return"].isnull().all():
            return {"status": "no_realized_outcomes_yet"}

        valid = df.dropna(subset=["predicted_direction", "actual_return"])
        if valid.empty:
            return {"status": "no_valid_records"}

        valid["actual_direction"] = (valid["actual_return"] > 0).astype(int)
        valid["correct"] = (valid["predicted_direction"] == valid["actual_direction"]).astype(int)

        accuracy = float(valid["correct"].mean())

        results = {
            "evaluated_samples": len(valid),
            "overall_directional_accuracy": accuracy,
            "last_updated": pd.Timestamp.now(tz="UTC").isoformat()
        }

        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump(results, f, indent=2)

        return results
