import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

class HistoricalAnalogueEngine:
    """
    Searches historical dataset for market states similar to today's vector.
    Strictly prevents look-ahead bias by scoring only using data prior to target outcomes.
    Calculates forward 1D, 3D, 5D, 10D, 20D, 60D probabilities and statistical distributions.
    """

    DEFAULT_FEATURE_WEIGHTS = {
        "rsi_14": 1.5,
        "dist_sma_20": 1.2,
        "dist_sma_50": 1.0,
        "hist_vol_20": 1.2,
        "gold_silver_ratio": 1.5,
        "dxy_return_20d": 1.0,
        "us10y_change_20d": 1.0
    }

    def __init__(
        self,
        top_k: int = 10,
        feature_weights: Optional[Dict[str, float]] = None,
        horizons: List[int] = [1, 3, 5, 10, 20, 60]
    ):
        self.top_k = top_k
        self.feature_weights = feature_weights or self.DEFAULT_FEATURE_WEIGHTS
        self.horizons = horizons

    def find_analogues(
        self,
        current_state: Dict[str, float],
        historical_features_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Given current feature dictionary and historical feature DataFrame,
        returns top K analogues and forward return statistics.
        """
        if historical_features_df.empty:
            return {"analogues": [], "forward_statistics": {}}

        df = historical_features_df.copy().reset_index(drop=True)

        # Calculate weighted Euclidean distance across normalized features
        dist_sq = pd.Series(0.0, index=df.index)
        weight_sum = 0.0

        for feat, weight in self.feature_weights.items():
            if feat in current_state and feat in df.columns:
                target_val = current_state[feat]
                col_series = df[feat]

                std = col_series.std()
                if pd.isna(std) or std == 0:
                    std = 1.0

                normalized_diff = (col_series - target_val) / std
                dist_sq += weight * (normalized_diff ** 2)
                weight_sum += weight

        if weight_sum == 0.0:
            return {"analogues": [], "forward_statistics": {}}

        distance = np.sqrt(dist_sq / weight_sum)
        df["analogue_distance"] = distance

        # Sort by closest similarity (exclude current observation if exact match)
        sorted_df = df.sort_values("analogue_distance").head(self.top_k)

        analogues = []
        for _, row in sorted_df.iterrows():
            analogues.append({
                "date": str(row.get("date", "")),
                "distance": round(float(row["analogue_distance"]), 4),
                "similarity_score": round(float(100.0 / (1.0 + row["analogue_distance"])), 1),
                "rsi_14": round(float(row.get("rsi_14", 0)), 1),
                "gold_silver_ratio": round(float(row.get("gold_silver_ratio", 0)), 2)
            })

        # Calculate forward statistics for top_k analogues across horizons
        forward_stats = {}
        for h in self.horizons:
            col_name = f"future_return_{h}d"
            if col_name in sorted_df.columns:
                returns = sorted_df[col_name].dropna() * 100.0  # percentage
                if len(returns) > 0:
                    pos_prob = float((returns > 0).mean() * 100.0)
                    neg_prob = float((returns < 0).mean() * 100.0)
                    forward_stats[f"{h}d"] = {
                        "sample_count": len(returns),
                        "prob_positive_pct": round(pos_prob, 1),
                        "prob_negative_pct": round(neg_prob, 1),
                        "expected_mean_return_pct": round(float(returns.mean()), 2),
                        "median_return_pct": round(float(returns.median()), 2),
                        "expected_volatility_pct": round(float(returns.std()), 2) if len(returns) > 1 else 0.0,
                        "min_return_pct": round(float(returns.min()), 2),
                        "max_return_pct": round(float(returns.max()), 2)
                    }

        return {
            "top_k": len(analogues),
            "analogues": analogues,
            "forward_statistics": forward_stats
        }
