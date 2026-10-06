import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

DISCLAIMER = "All outputs are probabilistic research signals derived from historical conditional statistics and do not represent guaranteed future price movements or financial advice."

class ForwardProbabilityEngine:
    """Generates probabilistic conditional forward return research signals for 1D, 3D, 5D, 10D, 20D, and 60D horizons."""

    @staticmethod
    def calculate_forward_probabilities(feature_df: pd.DataFrame, target_date: str, instrument: str = "GOLD") -> Dict[str, Any]:
        if feature_df.empty or "date" not in feature_df.columns:
            return {"disclaimer": DISCLAIMER, "horizons": {}}

        past_df = feature_df[feature_df["date"] < target_date].copy()
        target_row = feature_df[feature_df["date"] == target_date]

        if past_df.empty or target_row.empty:
            return {"disclaimer": DISCLAIMER, "horizons": {}}

        t_last = target_row.iloc[-1]
        curr_rsi = t_last.get("rsi_14", 50.0)

        # Condition history on similar momentum (RSI bucket +/- 10 points)
        conditioned = past_df[(past_df["rsi_14"] >= curr_rsi - 10) & (past_df["rsi_14"] <= curr_rsi + 10)]
        if len(conditioned) < 10:
            conditioned = past_df  # Fallback to full historical sample if regime sample too small

        horizons_stats = {}
        for h in [1, 3, 5, 10, 20, 60]:
            col_h = f"future_return_{h}d"
            if col_h in conditioned.columns:
                rets = conditioned[col_h].dropna()
                if not rets.empty:
                    pos_prob = float((rets > 0).mean())
                    neg_prob = float((rets < 0).mean())
                    horizons_stats[f"{h}d"] = {
                        "horizon_days": h,
                        "sample_count": len(rets),
                        "prob_positive": pos_prob,
                        "prob_negative": neg_prob,
                        "expected_return": float(rets.mean()),
                        "median_return": float(rets.median()),
                        "expected_volatility": float(rets.std()),
                        "historical_min": float(rets.min()),
                        "historical_max": float(rets.max())
                    }

        return {
            "date": target_date,
            "instrument": instrument,
            "disclaimer": DISCLAIMER,
            "conditioned_rsi_bucket": f"{curr_rsi - 10:.1f} - {curr_rsi + 10:.1f}",
            "horizons": horizons_stats
        }
