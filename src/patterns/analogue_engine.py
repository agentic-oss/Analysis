import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class AnalogueEngine:
    """
    Core historical analogue pattern-matching engine.
    Given a current feature vector, searches historical database for most similar dates using Euclidean / Cosine similarity.
    Calculates forward returns (1D, 3D, 5D, 10D, 20D, 60D) without look-ahead bias.
    """

    def __init__(self, feature_cols: List[str] = None):
        self.feature_cols = feature_cols or [
            "rsi_14", "macd", "volatility_20d", "distance_sma_50", "distance_sma_200",
            "gold_silver_ratio_zscore", "dxy_return_20d", "us10y_change_20d"
        ]

    def find_analogues(self, feature_df: pd.DataFrame, current_date: str, top_n: int = 5) -> Dict[str, Any]:
        """
        Finds top N historical analogues for the given date.
        """
        if feature_df.empty or current_date not in feature_df.index:
            # Fallback if exact date string index missing or empty
            if not feature_df.empty:
                curr_idx = len(feature_df) - 1
            else:
                return {}
        else:
            curr_idx = feature_df.index.get_loc(current_date)

        # Historical window: strictly before current date (no look-ahead bias)
        hist_df = feature_df.iloc[:curr_idx].copy()
        if len(hist_df) < 30:
            return {"warning": "Insufficient historical observations for analogue matching", "analogues": []}

        # Available matching features
        avail_cols = [c for c in self.feature_cols if c in feature_df.columns]
        if not avail_cols:
            return {"warning": "No matching feature columns found", "analogues": []}

        curr_vector = feature_df.iloc[curr_idx][avail_cols].values.astype(float)

        # Standardize features using historical mean/std
        hist_matrix = hist_df[avail_cols].values.astype(float)
        mean = np.nanmean(hist_matrix, axis=0)
        std = np.nanstd(hist_matrix, axis=0)
        std[std == 0] = 1.0

        curr_norm = np.nan_to_num((curr_vector - mean) / std)
        hist_norm = np.nan_to_num((hist_matrix - mean) / std)

        # Compute Euclidean distances
        distances = np.sqrt(np.sum((hist_norm - curr_norm) ** 2, axis=1))

        # Exclude recent 20 trading days to avoid overlapping market regime matches
        valid_indices = np.where(np.arange(len(hist_df)) < (curr_idx - 20))[0]
        if len(valid_indices) == 0:
            valid_indices = np.arange(len(hist_df))

        sorted_valid = valid_indices[np.argsort(distances[valid_indices])]
        top_indices = sorted_valid[:top_n]

        analogues = []
        forward_returns = {1: [], 3: [], 5: [], 10: [], 20: [], 60: []}

        full_price_col = "price" if "price" in feature_df.columns else ("Close" if "Close" in feature_df.columns else None)

        for idx in top_indices:
            match_date = hist_df.index[idx]
            sim_score = float(1.0 / (1.0 + distances[idx]))

            # Extract forward returns if target columns exist or compute from price
            fwd_stats = {}
            for h in [1, 3, 5, 10, 20, 60]:
                target_col = f"future_return_{h}d"
                if target_col in feature_df.columns:
                    val = feature_df.iloc[idx].get(target_col, np.nan)
                elif full_price_col and (idx + h) < len(feature_df):
                    p0 = feature_df.iloc[idx][full_price_col]
                    ph = feature_df.iloc[idx + h][full_price_col]
                    val = (ph - p0) / p0 if p0 > 0 else np.nan
                else:
                    val = np.nan

                fwd_stats[f"{h}d"] = float(val) if not np.isnan(val) else None
                if not np.isnan(val):
                    forward_returns[h].append(val)

            analogues.append({
                "date": str(match_date),
                "similarity_score": round(sim_score, 4),
                "distance": round(float(distances[idx]), 4),
                "forward_returns": fwd_stats
            })

        # Summary statistics across forward horizons
        horizon_summaries = {}
        for h, ret_list in forward_returns.items():
            if ret_list:
                arr = np.array(ret_list)
                horizon_summaries[f"{h}d"] = {
                    "sample_count": len(arr),
                    "mean_return_pct": float(round(np.mean(arr) * 100, 2)),
                    "median_return_pct": float(round(np.median(arr) * 100, 2)),
                    "prob_positive_pct": float(round((arr > 0).mean() * 100, 1)),
                    "prob_negative_pct": float(round((arr < 0).mean() * 100, 1)),
                    "max_gain_pct": float(round(np.max(arr) * 100, 2)),
                    "max_loss_pct": float(round(np.min(arr) * 100, 2))
                }
            else:
                horizon_summaries[f"{h}d"] = None

        return {
            "current_date": str(current_date),
            "top_analogues": analogues,
            "forward_horizon_statistics": horizon_summaries
        }
