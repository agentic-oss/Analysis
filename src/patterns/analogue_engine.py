"""
Historical Analogue Engine.
Finds most similar historical market states to today's state using normalized similarity distance,
and computes forward returns (1D, 3D, 5D, 10D, 20D, 60D) without look-ahead bias.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class HistoricalAnalogueEngine:
    """Calculates state similarity and evaluates forward returns on historical matches."""

    def __init__(
        self,
        feature_cols: List[str] = None,
        horizons: List[int] = [1, 3, 5, 10, 20, 60],
        top_k: int = 15,
    ):
        self.feature_cols = feature_cols or [
            "rsi_14",
            "macd_hist",
            "volatility_20d",
            "dist_sma_50_pct",
            "dist_sma_200_pct",
            "gold_silver_ratio",
            "ratio_zscore",
        ]
        self.horizons = horizons
        self.top_k = top_k

    def find_analogues(
        self,
        df: pd.DataFrame,
        current_idx: int,
        min_history_gap: int = 60,
    ) -> Dict[str, Any]:
        """
        Finds historical dates prior to current_idx - min_history_gap that have closest Euclidean
        distance on normalized features to current_idx.
        Strictly avoids look-ahead bias by only searching strictly past data.
        """
        if current_idx < min_history_gap + 30 or df.empty:
            return {"analogues": [], "forward_stats": {}}

        # Target current feature vector
        current_row = df.iloc[current_idx]
        available_features = [c for c in self.feature_cols if c in df.columns]

        if not available_features:
            return {"analogues": [], "forward_stats": {}}

        # Subset historical search space strictly in past
        hist_df = df.iloc[: current_idx - min_history_gap].copy()
        hist_df = hist_df.dropna(subset=available_features)

        if hist_df.empty:
            return {"analogues": [], "forward_stats": {}}

        # Convert feature columns strictly to float64
        for col in available_features:
            hist_df[col] = hist_df[col].astype(float)

        current_feats = current_row[available_features].astype(float)

        # Standardize features using historical statistics only
        means = hist_df[available_features].mean()
        stds = hist_df[available_features].std().replace(0, 1.0)

        current_norm = (current_feats - means) / stds
        hist_norm = (hist_df[available_features] - means) / stds

        # Euclidean distance using numpy arrays
        sq_diffs = ((hist_norm.to_numpy() - current_norm.to_numpy()) ** 2).sum(axis=1)
        dists = np.sqrt(sq_diffs)
        hist_df["similarity_distance"] = dists

        # Rank closest matches
        top_matches = hist_df.sort_values("similarity_distance").head(self.top_k)

        analogues_list = []
        forward_stats_by_horizon = {}

        for h in self.horizons:
            returns_h = []

            for idx, match_row in top_matches.iterrows():
                # Locate match index in full df to extract forward returns
                m_idx = df.index.get_loc(idx)
                if m_idx + h < len(df):
                    ret = (df.iloc[m_idx + h]["close"] - match_row["close"]) / match_row["close"]
                    returns_h.append(float(ret))
                    if h == 1:
                        analogues_list.append({
                            "date": str(match_row["date"]),
                            "close": float(match_row["close"]),
                            "distance": float(match_row["similarity_distance"]),
                        })

            if returns_h:
                returns_arr = np.array(returns_h, dtype=float)
                pos_prob = float((returns_arr > 0).mean())
                neg_prob = float((returns_arr < 0).mean())
                mean_ret = float(np.mean(returns_arr))
                median_ret = float(np.median(returns_arr))
                vol = float(np.std(returns_arr))
                max_gain = float(np.max(returns_arr))
                max_loss = float(np.min(returns_arr))

                forward_stats_by_horizon[f"{h}d"] = {
                    "sample_count": len(returns_arr),
                    "prob_positive": round(pos_prob, 4),
                    "prob_negative": round(neg_prob, 4),
                    "expected_return": round(mean_ret, 4),
                    "median_return": round(median_ret, 4),
                    "volatility": round(vol, 4),
                    "max_gain": round(max_gain, 4),
                    "max_loss": round(max_loss, 4),
                }

        return {
            "current_date": str(current_row["date"]),
            "analogues": analogues_list[: self.top_k],
            "forward_stats": forward_stats_by_horizon,
        }
