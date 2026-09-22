"""
Historical analogue search engine matching current market state to past historical observations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List


class HistoricalAnalogueEngine:
    """
    Finds historical analogue dates matching today's multi-factor market state without look-ahead bias.
    """

    def __init__(self, top_n: int = 10, min_history_gap_days: int = 30):
        self.top_n = top_n
        self.min_history_gap_days = min_history_gap_days

    def find_analogues(
        self,
        historical_df: pd.DataFrame,
        current_date: str,
        feature_cols: List[str] = [
            "rsi_14",
            "dist_sma_200_pct",
            "volatility_20d",
            "gold_silver_ratio",
            "dxy_return_20d",
            "us10y_change_20d",
        ],
    ) -> Dict[str, Any]:
        """
        Calculates similarity distances between target current_date observation and strictly prior historical observations.
        Returns top analogues and forward return statistics (1D, 3D, 5D, 10D, 20D, 60D).
        """
        if historical_df.empty or "date" not in historical_df.columns:
            return {"analogues": [], "forward_statistics": {}}

        df = historical_df.sort_values("date").reset_index(drop=True)

        if current_date not in df["date"].values:
            current_idx = len(df) - 1
            current_date = df.loc[current_idx, "date"]
        else:
            current_idx = df[df["date"] == current_date].index[0]

        # Strictest look-ahead protection
        max_valid_idx = current_idx - self.min_history_gap_days
        if max_valid_idx <= 0:
            return {"analogues": [], "forward_statistics": {}}

        prior_df = df.iloc[:max_valid_idx].copy()
        target_row = df.iloc[current_idx]

        valid_cols = [col for col in feature_cols if col in df.columns and col in prior_df.columns]
        if not valid_cols:
            return {"analogues": [], "forward_statistics": {}}

        # Ensure float types
        prior_matrix = prior_df[valid_cols].astype(float).values
        target_vector = target_row[valid_cols].astype(float).values

        means = np.nanmean(prior_matrix, axis=0)
        stds = np.nanstd(prior_matrix, axis=0)
        stds[stds == 0] = 1.0

        prior_scaled = (prior_matrix - means) / stds
        target_scaled = (target_vector - means) / stds

        # Euclidean Distance
        diffs = prior_scaled - target_scaled
        distances = np.sqrt(np.nansum(diffs ** 2, axis=1))

        prior_df["distance"] = distances
        prior_df["similarity_score"] = 100.0 / (1.0 + distances)

        top_analogues_df = prior_df.sort_values("distance").head(self.top_n)

        analogues_list = []
        for idx, row in top_analogues_df.iterrows():
            analogues_list.append({
                "date": row["date"],
                "similarity_score": round(float(row["similarity_score"]), 2),
                "distance": round(float(row["distance"]), 4),
                "macro_regime": str(row.get("macro_regime", "N/A")),
                "rsi_14": round(float(row.get("rsi_14", 0.0)), 2) if not pd.isna(row.get("rsi_14")) else None,
            })

        horizons = [1, 3, 5, 10, 20, 60]
        fwd_stats = {}

        for h in horizons:
            col_name = f"future_gold_ret_{h}d" if "future_gold_ret_1d" in df.columns else f"future_ret_{h}d"
            if col_name in top_analogues_df.columns:
                rets = top_analogues_df[col_name].dropna()
                if len(rets) > 0:
                    pos_pct = float((rets > 0).mean() * 100)
                    neg_pct = float((rets < 0).mean() * 100)
                    fwd_stats[f"{h}d"] = {
                        "sample_count": len(rets),
                        "mean_return_pct": float(round(rets.mean() * 100, 2)),
                        "median_return_pct": float(round(rets.median() * 100, 2)),
                        "positive_probability_pct": float(round(pos_pct, 1)),
                        "negative_probability_pct": float(round(neg_pct, 1)),
                        "max_gain_pct": float(round(rets.max() * 100, 2)),
                        "max_loss_pct": float(round(rets.min() * 100, 2)),
                    }

        return {
            "target_date": current_date,
            "analogues": analogues_list,
            "forward_statistics": fwd_stats,
        }
