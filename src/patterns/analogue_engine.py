import pandas as pd
import numpy as np
from typing import Dict, Any, List

class HistoricalAnalogueEngine:
    """Matches current market state against historical observations without look-ahead bias."""

    @staticmethod
    def find_analogues(
        df: pd.DataFrame,
        feature_cols: List[str],
        current_idx: int,
        top_k: int = 15,
        horizons: List[int] = [1, 3, 5, 10, 20, 60]
    ) -> Dict[str, Any]:
        """
        Calculates normalized Euclidean distance between the feature vector at current_idx
        and all historical rows prior to current_idx - max(horizons).
        Computes forward return statistics for the top-k matched historical dates.
        """
        max_horizon = max(horizons)
        if current_idx < max_horizon + 30 or current_idx >= len(df):
            return {"sample_count": 0, "analogues": [], "forward_stats": {}}

        # Ensure feature columns exist
        available_cols = [c for c in feature_cols if c in df.columns]
        if not available_cols:
            return {"sample_count": 0, "analogues": [], "forward_stats": {}}

        # Standardize features across historical window up to current_idx
        hist_df = df.iloc[:current_idx + 1].copy()
        feature_matrix = hist_df[available_cols].astype(float)

        # Fill missing values
        feature_matrix = feature_matrix.fillna(0.0)

        mean = feature_matrix.mean()
        std = feature_matrix.std().replace(0, 1.0)
        norm_matrix = (feature_matrix - mean) / std

        current_vector = norm_matrix.iloc[current_idx].values

        # Historical search space strictly prevents look-ahead bias: row i + max_horizon <= current_idx
        valid_hist_indices = range(30, current_idx - max_horizon)
        if not valid_hist_indices:
            return {"sample_count": 0, "analogues": [], "forward_stats": {}}

        distances = []
        for idx in valid_hist_indices:
            hist_vec = norm_matrix.iloc[idx].values
            dist = np.linalg.norm(current_vector - hist_vec)
            distances.append((idx, dist))

        # Sort by distance (smaller = more similar)
        distances.sort(key=lambda x: x[1])
        top_matches = distances[:top_k]

        analogues_list = []
        forward_returns_by_h = {h: [] for h in horizons}

        for idx, dist in top_matches:
            match_date = str(df.iloc[idx]['date'])
            similarity_score = round(100.0 / (1.0 + float(dist)), 2)

            fwd_dict = {}
            for h in horizons:
                if idx + h < len(df):
                    ret = float((df.iloc[idx + h]['close'] - df.iloc[idx]['close']) / df.iloc[idx]['close'] * 100)
                    forward_returns_by_h[h].append(ret)
                    fwd_dict[f"{h}d"] = round(ret, 2)

            analogues_list.append({
                "date": match_date,
                "similarity_score": similarity_score,
                "distance": round(float(dist), 4),
                "forward_returns": fwd_dict
            })

        # Summary forward statistics per horizon
        forward_stats = {}
        for h in horizons:
            rets = forward_returns_by_h[h]
            if rets:
                rets_arr = np.array(rets)
                forward_stats[f"{h}d"] = {
                    "count": len(rets),
                    "mean_return_pct": round(float(np.mean(rets_arr)), 2),
                    "median_return_pct": round(float(np.median(rets_arr)), 2),
                    "win_rate_pct": round(float((rets_arr > 0).mean() * 100), 1),
                    "loss_rate_pct": round(float((rets_arr < 0).mean() * 100), 1),
                    "max_gain_pct": round(float(np.max(rets_arr)), 2),
                    "max_loss_pct": round(float(np.min(rets_arr)), 2),
                    "volatility_pct": round(float(np.std(rets_arr)), 2)
                }

        return {
            "sample_count": len(analogues_list),
            "analogues": analogues_list,
            "forward_stats": forward_stats
        }
