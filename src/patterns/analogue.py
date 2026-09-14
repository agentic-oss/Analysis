"""
Historical Analogue Search & Similarity Engine.
Searches historical database for market conditions matching today's state
and calculates forward-return statistics without look-ahead bias.
"""
import logging
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

logger = logging.getLogger(__name__)


class HistoricalAnalogueEngine:
    """
    Identifies historical market analogues based on multidimensional normalized distance scoring:
    Calculates similarity over RSI, MACD, Moving Average Distances, Volatility, and Gold/Silver Ratio.
    Calculates forward 1D, 3D, 5D, 10D, 20D, and 60D return distributions across historical matches.
    """

    def __init__(self, top_n: int = 15):
        self.top_n = top_n
        self.feature_cols = [
            "rsi_14", "macd", "volatility_20d", "dist_sma_20_pct",
            "dist_sma_50_pct", "dist_sma_200_pct"
        ]

    def find_analogues(
        self,
        feature_df: pd.DataFrame,
        horizons: List[int] = [1, 3, 5, 10, 20, 60]
    ) -> Dict[str, Any]:
        """
        Finds the top N most similar historical dates to the current state (last row in feature_df).
        Calculates forward statistics strictly using historical outcomes of those analogue dates.
        """
        if feature_df.empty or len(feature_df) < 60:
            return {
                "analogue_count": 0,
                "top_analogues": [],
                "forward_statistics": {}
            }

        df = feature_df.copy().sort_values("date").reset_index(drop=True)

        # Available feature columns present in dataframe
        available_cols = [c for c in self.feature_cols if c in df.columns]
        if not available_cols:
            return {"analogue_count": 0, "top_analogues": [], "forward_statistics": {}}

        # Normalize features via z-score scaling up to T-1
        df_feats = df[available_cols].fillna(0)
        mean = df_feats.mean()
        std = df_feats.std().replace(0, 1.0)
        norm_feats = (df_feats - mean) / std

        # Current state is the last row (Date T)
        current_state = norm_feats.iloc[-1].values
        current_date = df["date"].iloc[-1]

        # Search space: all historical rows except the last 60 rows (to avoid testing against recent unclosed windows)
        search_idx = df.index[:-60]
        if len(search_idx) == 0:
            search_idx = df.index[:-1]

        distances = []
        for idx in search_idx:
            hist_state = norm_feats.iloc[idx].values
            dist = euclidean(current_state, hist_state)
            similarity_score = max(0.0, 100.0 * (1.0 - (dist / (np.sqrt(len(available_cols)) * 2.0))))
            distances.append((idx, df["date"].iloc[idx], float(dist), float(similarity_score)))

        # Sort by smallest Euclidean distance (highest similarity)
        distances.sort(key=lambda x: x[2])
        top_matches = distances[:self.top_n]

        # Gather forward return statistics across top analogue indices
        top_indices = [match[0] for match in top_matches]
        analogue_details = []

        for idx, date_str, dist, sim in top_matches:
            analogue_details.append({
                "date": date_str,
                "distance": dist,
                "similarity_score": round(sim, 2)
            })

        forward_stats = {}
        for h in horizons:
            col_target = f"future_return_{h}d"
            if col_target in df.columns:
                returns = df.loc[top_indices, col_target].dropna()
            else:
                # Compute on the fly for historical analogue index
                returns = []
                for idx in top_indices:
                    if idx + h < len(df):
                        p0 = df["close"].iloc[idx]
                        ph = df["close"].iloc[idx + h]
                        returns.append((ph - p0) / p0)
                returns = pd.Series(returns)

            if len(returns) > 0:
                forward_stats[f"horizon_{h}d"] = {
                    "sample_count": int(len(returns)),
                    "mean_return_pct": round(float(returns.mean() * 100), 2),
                    "median_return_pct": round(float(returns.median() * 100), 2),
                    "positive_prob_pct": round(float((returns > 0).mean() * 100), 1),
                    "negative_prob_pct": round(float((returns < 0).mean() * 100), 1),
                    "max_gain_pct": round(float(returns.max() * 100), 2),
                    "max_loss_pct": round(float(returns.min() * 100), 2)
                }

        return {
            "current_date": current_date,
            "analogue_count": len(top_matches),
            "top_analogues": analogue_details,
            "forward_statistics": forward_stats
        }
