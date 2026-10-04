import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

class AnalogueEngine:
    """Historical Analogue Engine: Finds historical market states similar to current conditions and computes forward return statistics without look-ahead bias."""

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        top_k: int = 20,
        min_sample: int = 10
    ):
        self.top_k = top_k
        self.min_sample = min_sample
        self.feature_cols = feature_cols or [
            "rsi_14", "macd", "volatility_20d", "dist_sma_50", "dist_sma_200",
            "gold_silver_ratio_zscore", "dxy_return_20d", "us10y_change_20d", "vix_level"
        ]

    def find_analogues(
        self,
        historical_df: pd.DataFrame,
        current_state: pd.Series,
        horizons: List[int] = [1, 3, 5, 10, 20, 60]
    ) -> Dict[str, Any]:
        """Search historical feature database up to current date (excluding future info) to find closest analogue dates."""
        if historical_df.empty or len(historical_df) < self.min_sample:
            return {
                "sample_count": 0,
                "analogues": [],
                "forward_statistics": {}
            }

        avail_cols = [c for c in self.feature_cols if c in historical_df.columns and c in current_state.index]
        if not avail_cols:
            return {
                "sample_count": 0,
                "analogues": [],
                "forward_statistics": {}
            }

        # Normalize features (z-score scaling based on historical distribution)
        hist_feats = historical_df[avail_cols].astype(float).fillna(0.0)
        curr_feats = current_state[avail_cols].astype(float).fillna(0.0)

        mean = hist_feats.mean()
        std = hist_feats.std().replace(0, 1.0)

        norm_hist = (hist_feats - mean) / std
        norm_curr = (curr_feats - mean) / std

        # Compute Euclidean distance
        distances = np.sqrt(((norm_hist - norm_curr) ** 2).sum(axis=1))

        # Sort and take top_k (excluding exact current date if present in historical_df)
        hist_copy = historical_df.copy()
        hist_copy["distance"] = distances

        # Exclude exact current date match if same row
        if "date" in current_state.index and "date" in hist_copy.columns:
            hist_copy = hist_copy[hist_copy["date"] != current_state["date"]]

        top_analogues = hist_copy.sort_values("distance").head(self.top_k)

        # Calculate forward return statistics for top analogues across horizons
        forward_stats = {}
        for h in horizons:
            target_col = f"future_return_{h}d"
            if target_col in top_analogues.columns:
                returns = top_analogues[target_col].dropna().astype(float)
                if len(returns) > 0:
                    forward_stats[f"{h}d"] = {
                        "sample_count": len(returns),
                        "mean_return": float(returns.mean()),
                        "median_return": float(returns.median()),
                        "prob_positive": float((returns > 0).mean()),
                        "prob_negative": float((returns < 0).mean()),
                        "volatility": float(returns.std()) if len(returns) > 1 else 0.0,
                        "min_return": float(returns.min()),
                        "max_return": float(returns.max())
                    }

        # Format analogue records
        analogue_list = []
        for _, row in top_analogues.iterrows():
            rec = {
                "date": str(row.get("date", "")),
                "similarity_score": round(100.0 / (1.0 + float(row["distance"])), 2),
                "distance": float(row["distance"])
            }
            for h in horizons:
                t_col = f"future_return_{h}d"
                if t_col in row:
                    rec[f"return_{h}d"] = float(row[t_col]) if pd.notnull(row[t_col]) else None
            analogue_list.append(rec)

        return {
            "sample_count": len(top_analogues),
            "analogues": analogue_list,
            "forward_statistics": forward_stats
        }
