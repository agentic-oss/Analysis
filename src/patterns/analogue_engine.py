import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

class HistoricalAnalogueEngine:
    """Finds historical market analogues based on technical and macro similarity without look-ahead bias."""

    def __init__(self, horizons: List[int] = [1, 3, 5, 10, 20, 60]):
        self.horizons = horizons

    def find_analogues(
        self,
        df: pd.DataFrame,
        current_idx: int,
        price_col: str = "GOLD_close",
        top_n: int = 10,
        min_history: int = 100
    ) -> Dict[str, Any]:
        """
        Finds the top_n historical days (prior to current_idx) most similar to the state at current_idx.
        Computes forward returns for horizons strictly ensuring analogue index + horizon <= current_idx
        to prevent look-ahead bias during historical evaluation.
        """
        if current_idx < min_history or current_idx >= len(df):
            return {"sample_count": 0, "analogues": [], "forward_stats": {}}

        current_row = df.iloc[current_idx]

        # Feature normalization across historical slice (0 to current_idx)
        hist_df = df.iloc[:current_idx].copy()
        if len(hist_df) < 20:
            return {"sample_count": 0, "analogues": [], "forward_stats": {}}

        # Selected features for distance scoring
        candidate_cols = [
            c for c in hist_df.columns
            if "rsi" in c or "stoch_rsi" in c or "dist_sma" in c or "ratio_zscore" in c or "historical_vol" in c
        ]

        if not candidate_cols:
            candidate_cols = [c for c in [price_col] if c in hist_df.columns]

        # Standardize features using current slice stats
        hist_features = hist_df[candidate_cols].astype(float)
        curr_features = current_row[candidate_cols].astype(float)

        mean = hist_features.mean()
        std = hist_features.std().replace(0, 1.0)

        hist_norm = (hist_features - mean) / std
        curr_norm = (curr_features - mean) / std

        # Compute Euclidean distance
        distances = np.sqrt(((hist_norm - curr_norm) ** 2).sum(axis=1))

        # Filter out observations that do not have full horizon history within current_idx
        max_horizon = max(self.horizons)
        valid_indices = [idx for idx in distances.index if idx + max_horizon <= current_idx]

        if not valid_indices:
            # Fallback if history is too short for max_horizon
            valid_indices = [idx for idx in distances.index if idx + 1 <= current_idx]

        if not valid_indices:
            return {"sample_count": 0, "analogues": [], "forward_stats": {}}

        sorted_analogues = distances.loc[valid_indices].sort_values().head(top_n)

        analogues_list = []
        forward_returns = {f"{h}d": [] for h in self.horizons}

        p_series = df[price_col].astype(float)

        for idx, dist in sorted_analogues.items():
            analogue_date = df.loc[idx, "date"]
            analogue_p = p_series.loc[idx]

            returns_by_h = {}
            for h in self.horizons:
                if idx + h <= current_idx:
                    fwd_p = p_series.iloc[idx + h]
                    ret = (fwd_p - analogue_p) / analogue_p
                    forward_returns[f"{h}d"].append(ret)
                    returns_by_h[f"{h}d"] = float(ret)

            analogues_list.append({
                "date": str(analogue_date),
                "distance": float(dist),
                "price": float(analogue_p),
                "returns": returns_by_h
            })

        # Calculate forward return statistics
        stats = {}
        for h_str, rets in forward_returns.items():
            if rets:
                r = np.array(rets)
                stats[h_str] = {
                    "count": len(r),
                    "mean_return": float(np.mean(r)),
                    "median_return": float(np.median(r)),
                    "pos_prob": float((r > 0).mean()),
                    "neg_prob": float((r < 0).mean()),
                    "max_gain": float(np.max(r)),
                    "max_loss": float(np.min(r)),
                    "std": float(np.std(r))
                }

        return {
            "current_date": str(current_row["date"]),
            "sample_count": len(analogues_list),
            "analogues": analogues_list,
            "forward_stats": stats
        }
