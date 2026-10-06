import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class HistoricalAnalogueEngine:
    """Searches historical dataset for market states similar to current state without look-ahead bias and calculates forward return distributions."""

    @staticmethod
    def find_analogues(feature_df: pd.DataFrame, target_date: str, top_k: int = 10) -> Dict[str, Any]:
        if feature_df.empty or "date" not in feature_df.columns:
            return {"sample_count": 0, "analogues": [], "forward_stats": {}}

        # Strict temporal split: only look at history prior to target_date
        past_df = feature_df[feature_df["date"] < target_date].copy()
        target_row = feature_df[feature_df["date"] == target_date]

        if past_df.empty or target_row.empty:
            return {"sample_count": 0, "analogues": [], "forward_stats": {}}

        t_last = target_row.iloc[-1]

        # Features used for similarity comparison
        sim_cols = [col for col in ["rsi_14", "macd_hist", "dist_sma_20", "dist_sma_200", "volatility_20d", "gold_silver_ratio"] if col in past_df.columns]

        if not sim_cols:
            return {"sample_count": 0, "analogues": [], "forward_stats": {}}

        # Standardize features using past statistics
        means = past_df[sim_cols].mean()
        stds = past_df[sim_cols].std().replace(0, 1.0)

        past_scaled = (past_df[sim_cols] - means) / stds
        target_scaled = (t_last[sim_cols] - means) / stds

        # Euclidean distance similarity score (inverse of distance)
        dist_sq = ((past_scaled - target_scaled) ** 2).sum(axis=1, numeric_only=True)
        distances = pd.Series(np.sqrt(dist_sq.to_numpy()), index=dist_sq.index)
        similarity_scores = 100.0 / (1.0 + distances)

        past_df = past_df.copy()
        past_df["similarity_score"] = similarity_scores
        top_matches = past_df.sort_values(by="similarity_score", ascending=False).head(top_k)

        analogues_list = []
        for idx, row in top_matches.iterrows():
            analogues_list.append({
                "date": str(row["date"]),
                "similarity_score": float(row["similarity_score"]),
                "rsi_14": float(row.get("rsi_14", 50.0)),
                "forward_1d": float(row.get("future_return_1d", 0.0) or 0.0),
                "forward_5d": float(row.get("future_return_5d", 0.0) or 0.0),
                "forward_10d": float(row.get("future_return_10d", 0.0) or 0.0),
                "forward_20d": float(row.get("future_return_20d", 0.0) or 0.0),
                "forward_60d": float(row.get("future_return_60d", 0.0) or 0.0)
            })

        # Calculate forward stats across top analogues
        forward_stats = {}
        for h in [1, 3, 5, 10, 20, 60]:
            col_h = f"future_return_{h}d"
            if col_h in top_matches.columns:
                rets = top_matches[col_h].dropna()
                if not rets.empty:
                    forward_stats[f"{h}d"] = {
                        "sample_count": len(rets),
                        "mean_return": float(rets.mean()),
                        "median_return": float(rets.median()),
                        "prob_positive": float((rets > 0).mean()),
                        "prob_negative": float((rets < 0).mean()),
                        "max_gain": float(rets.max()),
                        "max_loss": float(rets.min())
                    }

        return {
            "target_date": target_date,
            "sample_count": len(top_matches),
            "top_k": top_k,
            "analogues": analogues_list,
            "forward_stats": forward_stats
        }
