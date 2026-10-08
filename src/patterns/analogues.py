"""
Historical Analogue Engine.
Searches historical feature store for market state similarity and computes forward returns.
"""
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np


class AnalogueEngine:
    """
    Finds top historical analogues for today's market state based on normalized features (RSI, MACD, Volatility, MAs distance, etc.)
    and calculates forward returns distributions (1D, 3D, 5D, 10D, 20D, 60D).
    """

    @staticmethod
    def find_analogues(
        historical_features_df: pd.DataFrame,
        current_feature_row: pd.Series,
        top_n: int = 10,
        feature_cols: Optional[List[str]] = None,
        horizons: List[int] = [1, 3, 5, 10, 20, 60],
    ) -> Dict[str, Any]:
        """
        Finds top_n historical records closest in Euclidean distance to current_feature_row without look-ahead bias.
        """
        if historical_features_df.empty or current_feature_row.empty:
            return {}

        if feature_cols is None:
            feature_cols = [
                "rsi_14",
                "stoch_rsi",
                "macd_hist",
                "volatility_20d",
                "dist_sma_20",
                "dist_sma_50",
                "dist_sma_200",
            ]

        # Filter features that exist in both
        available_cols = [c for c in feature_cols if c in historical_features_df.columns and c in current_feature_row.index]
        if not available_cols:
            return {}

        hist_df = historical_features_df.dropna(subset=available_cols).copy()

        # Exclude recent rows to prevent autocorrelation / current day overlap
        if len(hist_df) > 60:
            hist_df = hist_df.iloc[:-60]

        if hist_df.empty:
            return {}

        # Convert to float float64 array for clean numeric math
        hist_vals = hist_df[available_cols].astype(float)
        curr_vals = current_feature_row[available_cols].astype(float)

        # Normalize features (Z-score standard scale)
        mean_vals = hist_vals.mean()
        std_vals = hist_vals.std().replace(0, 1.0)

        hist_scaled = (hist_vals - mean_vals) / std_vals
        curr_scaled = (curr_vals - mean_vals) / std_vals

        # Compute Euclidean distance using float array
        diff_sq = (hist_scaled - curr_scaled) ** 2
        distances = np.sqrt(diff_sq.sum(axis=1).astype(float))
        hist_df["similarity_distance"] = distances

        # Top N nearest neighbors
        top_analogues = hist_df.sort_values("similarity_distance").head(top_n)

        # Calculate forward statistics for top analogues across horizons
        forward_stats = {}
        for h in horizons:
            col_target = f"future_return_{h}d"
            if col_target in top_analogues.columns:
                rets = pd.to_numeric(top_analogues[col_target], errors="coerce").dropna()
                if not rets.empty:
                    pos_prob = float((rets > 0).mean() * 100.0)
                    mean_ret = float(rets.mean() * 100.0)
                    median_ret = float(rets.median() * 100.0)
                    max_gain = float(rets.max() * 100.0)
                    max_loss = float(rets.min() * 100.0)

                    forward_stats[f"{h}d"] = {
                        "sample_count": len(rets),
                        "positive_probability_pct": pos_prob,
                        "negative_probability_pct": 100.0 - pos_prob,
                        "mean_return_pct": mean_ret,
                        "median_return_pct": median_ret,
                        "max_gain_pct": max_gain,
                        "max_loss_pct": max_loss,
                    }

        # Build list of analogue dates
        analogue_matches = []
        for idx, row in top_analogues.iterrows():
            analogue_matches.append({
                "date": str(row["date"]),
                "distance": float(row["similarity_distance"]),
                "return_5d": float(row.get("future_return_5d", 0.0) * 100.0) if "future_return_5d" in row and not pd.isna(row.get("future_return_5d")) else None,
                "return_20d": float(row.get("future_return_20d", 0.0) * 100.0) if "future_return_20d" in row and not pd.isna(row.get("future_return_20d")) else None,
            })

        return {
            "num_analogues_found": len(top_analogues),
            "top_analogues": analogue_matches,
            "forward_statistics": forward_stats,
        }
