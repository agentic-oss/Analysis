import numpy as np
import pandas as pd
from typing import Dict, Any, List


class HistoricalAnalogueEngine:
    """Finds historical market states similar to the current market state and computes forward return statistics."""

    def __init__(self, top_k: int = 20, similarity_threshold: float = 0.5):
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def find_analogues(
        self,
        historical_features_df: pd.DataFrame,
        current_state: Dict[str, float],
        feature_cols: List[str],
        horizons: List[int] = [1, 3, 5, 10, 20, 60],
    ) -> Dict[str, Any]:
        """Compares current state vector against historical dataset without look-ahead bias."""
        if historical_features_df.empty or len(historical_features_df) < 30:
            return {
                "top_analogues": [],
                "forward_statistics": {f"{h}d": self._empty_stats() for h in horizons},
            }

        df = historical_features_df.copy().sort_values("date").reset_index(drop=True)

        # Exclude recent rows that do not have future outcomes available for evaluation
        # Calculate standard distance across feature_cols
        valid_cols = [c for c in feature_cols if c in df.columns and c in current_state]
        if not valid_cols:
            return {
                "top_analogues": [],
                "forward_statistics": {f"{h}d": self._empty_stats() for h in horizons},
            }

        # Normalize features (z-score)
        feature_matrix = df[valid_cols].values
        curr_vec = np.array([current_state[c] for c in valid_cols])

        means = np.nanmean(feature_matrix, axis=0)
        stds = np.nanstd(feature_matrix, axis=0) + 1e-8

        norm_matrix = (feature_matrix - means) / stds
        norm_curr = (curr_vec - means) / stds

        # Euclidean distances & normalized similarity score (0 to 1)
        distances = np.sqrt(np.nansum((norm_matrix - norm_curr) ** 2, axis=1))
        max_dist = np.nanmax(distances) + 1e-8
        similarity_scores = 1.0 - (distances / max_dist)

        df["similarity_score"] = similarity_scores

        # Sort historical observations by similarity, excluding the current/most recent day
        historical_candidates = df.iloc[:-1].sort_values("similarity_score", ascending=False)
        top_matches = historical_candidates[historical_candidates["similarity_score"] >= self.similarity_threshold].head(self.top_k)

        if top_matches.empty:
            top_matches = historical_candidates.head(self.top_k)

        top_analogues_list = []
        for idx, row in top_matches.iterrows():
            top_analogues_list.append({
                "date": str(row["date"]),
                "similarity_score": round(float(row["similarity_score"]), 4),
                "close_price": round(float(row["close"]), 2) if "close" in row else None,
            })

        # Calculate forward return statistics for matches
        forward_stats = {}
        for h in horizons:
            col_target = f"future_return_{h}d"
            if col_target in top_matches.columns:
                returns = top_matches[col_target].dropna().values
            else:
                returns = np.array([])

            forward_stats[f"{h}d"] = self._compute_horizon_stats(returns)

        return {
            "top_analogues": top_analogues_list,
            "forward_statistics": forward_stats,
        }

    def _compute_horizon_stats(self, returns: np.ndarray) -> Dict[str, Any]:
        if len(returns) == 0:
            return self._empty_stats()

        pos_prob = float(np.mean(returns > 0))
        neg_prob = float(np.mean(returns < 0))
        mean_ret = float(np.mean(returns))
        median_ret = float(np.median(returns))
        volatility = float(np.std(returns))
        max_gain = float(np.max(returns))
        max_loss = float(np.min(returns))

        return {
            "sample_count": len(returns),
            "prob_positive": round(pos_prob, 4),
            "prob_negative": round(neg_prob, 4),
            "mean_return": round(mean_ret, 6),
            "median_return": round(median_ret, 6),
            "expected_volatility": round(volatility, 6),
            "max_gain": round(max_gain, 6),
            "max_loss": round(max_loss, 6),
        }

    def _empty_stats(self) -> Dict[str, Any]:
        return {
            "sample_count": 0,
            "prob_positive": 0.5,
            "prob_negative": 0.5,
            "mean_return": 0.0,
            "median_return": 0.0,
            "expected_volatility": 0.0,
            "max_gain": 0.0,
            "max_loss": 0.0,
        }
