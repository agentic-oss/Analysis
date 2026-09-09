"""
Historical Analogue Engine.
Searches historical database for observations most similar to today's multi-factor market state
(RSI, MACD, MAs, Volatility, G/S Ratio, DXY, Yields, VIX, Macro Regime).
Calculates forward 1D, 3D, 5D, 10D, 20D, 60D return distributions without look-ahead bias.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class AnalogueEngine:
    """Multi-factor historical similarity search and forward-return distribution engine."""

    def __init__(self, feature_weights: Optional[Dict[str, float]] = None):
        self.feature_weights = feature_weights or {
            "rsi_14": 1.5,
            "dist_sma_50_pct": 1.2,
            "volatility_20d": 1.0,
            "gold_silver_ratio": 1.5,
            "dxy_return_20d": 1.0,
            "us10y_change_20d": 1.0,
            "vix_level": 1.0
        }

    def find_analogues(
        self,
        feature_df: pd.DataFrame,
        target_instrument: str = "GOLD",
        top_k: int = 15,
        min_history_buffer_days: int = 65
    ) -> Dict[str, Any]:
        """
        Finds top_k most similar historical observations to current date T.
        Ensures strict temporal isolation: historical observations used must exclude recent dates [T - min_history_buffer_days, T].
        """
        if feature_df.empty or len(feature_df) < min_history_buffer_days + 50:
            return {"top_analogues": [], "forward_return_statistics": {}}

        df = feature_df.copy().sort_values("date").reset_index(drop=True)
        current_idx = len(df) - 1
        current_row = df.iloc[current_idx]

        # Candidate historical search pool excluding look-ahead window and recent lookback window
        valid_pool_end = max(0, current_idx - min_history_buffer_days)
        pool = df.iloc[:valid_pool_end].copy()

        if pool.empty:
            return {"top_analogues": [], "forward_return_statistics": {}}

        # Normalize features across pool + current observation to calculate Euclidean distances
        avail_features = [f for f in self.feature_weights.keys() if f in df.columns]
        if not avail_features:
            return {"top_analogues": [], "forward_return_statistics": {}}

        feature_matrix = df[avail_features].copy().fillna(0.0)
        means = feature_matrix.mean()
        stds = feature_matrix.std().replace(0, 1.0)
        norm_matrix = (feature_matrix - means) / stds

        curr_vec = norm_matrix.iloc[current_idx].values
        pool_indices = pool.index

        distances = []
        for p_idx in pool_indices:
            p_vec = norm_matrix.iloc[p_idx].values
            # Weighted Euclidean distance
            w_vec = np.array([self.feature_weights[f] for f in avail_features])
            dist = np.sqrt(np.sum(w_vec * ((curr_vec - p_vec) ** 2)))
            distances.append((p_idx, dist))

        distances.sort(key=lambda x: x[1])
        top_matches = distances[:top_k]

        analogue_records = []
        fwd_returns_by_horizon = {h: [] for h in [1, 3, 5, 10, 20, 60]}

        for match_idx, dist in top_matches:
            match_row = df.iloc[match_idx]
            sim_score = float(max(0.0, 100.0 - dist * 10.0))

            item = {
                "historical_date": str(match_row["date"]),
                "distance": float(dist),
                "similarity_score": sim_score,
                "price": float(match_row.get("close", 0.0)),
                "rsi_14": float(match_row.get("rsi_14", 50.0)),
                "forward_returns": {}
            }

            for h in [1, 3, 5, 10, 20, 60]:
                ret_col = f"future_return_{h}d"
                if ret_col in match_row and pd.notna(match_row[ret_col]):
                    val = float(match_row[ret_col])
                    item["forward_returns"][f"{h}d"] = val
                    fwd_returns_by_horizon[h].append(val)

            analogue_records.append(item)

        # Aggregate statistics
        fwd_stats = {}
        for h, rets in fwd_returns_by_horizon.items():
            if rets:
                r_arr = np.array(rets)
                fwd_stats[f"{h}d"] = {
                    "sample_count": len(r_arr),
                    "mean_return": float(np.mean(r_arr)),
                    "median_return": float(np.median(r_arr)),
                    "positive_prob_pct": float((r_arr > 0).mean() * 100.0),
                    "negative_prob_pct": float((r_arr < 0).mean() * 100.0),
                    "max_gain": float(np.max(r_arr)),
                    "max_loss": float(np.min(r_arr)),
                    "percentile_25": float(np.percentile(r_arr, 25)),
                    "percentile_75": float(np.percentile(r_arr, 75))
                }

        return {
            "as_of_date": str(current_row["date"]),
            "target_instrument": target_instrument,
            "top_analogues": analogue_records,
            "forward_return_statistics": fwd_stats
        }
