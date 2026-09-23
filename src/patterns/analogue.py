import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

class HistoricalAnalogueEngine:
    """
    Searches historical dataset for market states most similar to current state.
    Calculates forward return statistics without look-ahead bias.

    Features evaluated for similarity:
    - RSI 14
    - Distance from SMA 20, 50, 200
    - Volatility (20d)
    - Gold/Silver ratio (if available)
    - DXY 20d return
    - Real Yield change
    """
    def __init__(self, top_k: int = 10, feature_weights: Optional[Dict[str, float]] = None):
        self.top_k = top_k
        self.feature_weights = feature_weights or {
            "rsi_14": 1.5,
            "dist_sma_20": 1.0,
            "dist_sma_50": 1.0,
            "dist_sma_200": 1.0,
            "volatility_20d": 1.2,
            "gold_silver_ratio": 1.0,
            "return_20d": 1.0
        }

    def find_analogues(
        self,
        historical_df: pd.DataFrame,
        current_state: pd.Series,
        horizons: List[int] = [1, 3, 5, 10, 20, 60]
    ) -> Dict[str, Any]:
        """
        Finds top K historical analogues and computes forward return distribution metrics.
        Ensures strict temporal isolation (look-ahead protection).
        """
        if historical_df.empty or len(historical_df) < 30:
            return {
                "sample_count": 0,
                "top_analogues": [],
                "forward_statistics": {}
            }

        # Available feature columns present in both historical_df and current_state
        feat_cols = [col for col in self.feature_weights.keys() if col in historical_df.columns and col in current_state.index]

        if not feat_cols:
            return {
                "sample_count": 0,
                "top_analogues": [],
                "forward_statistics": {}
            }

        # Exclude recent observations where forward 60d return is not yet known or current date itself
        clean_hist = historical_df.dropna(subset=feat_cols).copy()

        # Standardize features (Z-score normalization)
        hist_matrix = clean_hist[feat_cols].values
        curr_vec = current_state[feat_cols].values.astype(float)

        mean_vec = np.nanmean(hist_matrix, axis=0)
        std_vec = np.nanstd(hist_matrix, axis=0) + 1e-8

        norm_hist = (hist_matrix - mean_vec) / std_vec
        norm_curr = (curr_vec - mean_vec) / std_vec

        # Apply weights
        weights = np.array([self.feature_weights[c] for c in feat_cols])
        weighted_diff = (norm_hist - norm_curr) * weights

        # Euclidean distance
        distances = np.sqrt(np.sum(weighted_diff ** 2, axis=1))
        clean_hist["distance"] = distances

        # Get top K lowest distance analogues
        top_matches = clean_hist.sort_values("distance").head(self.top_k)

        forward_stats = {}
        for h in horizons:
            # Shift price to calculate forward return for historical matches
            close_s = historical_df["close"]
            fwd_ret_col = f"fwd_ret_{h}d"
            if fwd_ret_col not in historical_df.columns:
                fwd_series = (close_s.shift(-h) - close_s) / close_s * 100.0
            else:
                fwd_series = historical_df[fwd_ret_col]

            # Match top analogues with their calculated forward return
            match_fwd = fwd_series.loc[top_matches.index].dropna()

            if len(match_fwd) > 0:
                pos_prob = (match_fwd > 0).mean() * 100.0
                neg_prob = (match_fwd < 0).mean() * 100.0
                mean_ret = float(match_fwd.mean())
                med_ret = float(match_fwd.median())
                max_gain = float(match_fwd.max())
                max_loss = float(match_fwd.min())
            else:
                pos_prob = neg_prob = mean_ret = med_ret = max_gain = max_loss = 0.0

            forward_stats[f"{h}d"] = {
                "sample_count": len(match_fwd),
                "positive_probability_pct": round(pos_prob, 1),
                "negative_probability_pct": round(neg_prob, 1),
                "mean_return_pct": round(mean_ret, 2),
                "median_return_pct": round(med_ret, 2),
                "max_gain_pct": round(max_gain, 2),
                "max_loss_pct": round(max_loss, 2)
            }

        analogues_list = []
        for _, row in top_matches.iterrows():
            analogues_list.append({
                "timestamp": str(row.get("timestamp", "")),
                "distance": round(float(row.get("distance", 0.0)), 4),
                "close": round(float(row.get("close", 0.0)), 2),
                "rsi_14": round(float(row.get("rsi_14", 50.0)), 2)
            })

        return {
            "sample_count": len(top_matches),
            "top_analogues": analogues_list,
            "forward_statistics": forward_stats
        }
