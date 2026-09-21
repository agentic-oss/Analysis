import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple


class HistoricalAnalogueEngine:
    """Historical analogue search engine based on normalized state vector similarity."""

    FEATURE_COLS = [
        "rsi_14", "macd_hist", "dist_sma_20", "dist_sma_50",
        "dist_sma_200", "volatility_20d", "return_5d", "return_20d"
    ]

    @staticmethod
    def find_analogues(
        df: pd.DataFrame,
        top_n: int = 15,
        min_lookback_days: int = 60
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if df is None or df.empty or len(df) < min_lookback_days + 60:
            return [], {}

        df_sorted = df.sort_values("date").reset_index(drop=True)

        # Check required columns
        available_cols = [c for c in HistoricalAnalogueEngine.FEATURE_COLS if c in df_sorted.columns]
        if len(available_cols) < 4:
            return [], {}

        # Fill NaNs
        feature_matrix = df_sorted[available_cols].ffill().bfill().values

        # Normalize features (z-score scaling across historical range)
        f_mean = np.mean(feature_matrix[:-min_lookback_days], axis=0, keepdims=True)
        f_std = np.std(feature_matrix[:-min_lookback_days], axis=0, keepdims=True) + 1e-8

        norm_matrix = (feature_matrix - f_mean) / f_std

        # Current state vector (latest row)
        current_vector = norm_matrix[-1]

        # Historical pool (excluding recent min_lookback_days to avoid forward bias/overlap)
        hist_matrix = norm_matrix[:-min_lookback_days]
        hist_dates = df_sorted["date"].values[:-min_lookback_days]

        # Euclidean distances
        distances = np.linalg.norm(hist_matrix - current_vector, axis=1)

        # Convert distance to similarity score (0 to 100)
        similarity_scores = 100.0 / (1.0 + distances)

        # Top N indices
        top_indices = np.argsort(distances)[:top_n]

        top_analogues = []
        forward_returns = {h: [] for h in [1, 3, 5, 10, 20, 60]}

        for idx in top_indices:
            a_date = hist_dates[idx]
            sim_score = similarity_scores[idx]
            a_price = float(df_sorted.loc[idx, "close"])

            outcomes = {}
            for h in [1, 3, 5, 10, 20, 60]:
                if idx + h < len(df_sorted):
                    fwd_price = float(df_sorted.loc[idx + h, "close"])
                    fwd_ret = (fwd_price - a_price) / a_price
                    outcomes[f"{h}d"] = round(fwd_ret * 100.0, 2)
                    forward_returns[h].append(fwd_ret)

            top_analogues.append({
                "date": str(a_date),
                "similarity_score": round(float(sim_score), 1),
                "historical_price": round(a_price, 2),
                "forward_outcomes": outcomes
            })

        # Calculate statistics across analogues
        statistics = {"sample_count": len(top_analogues)}
        for h in [1, 3, 5, 10, 20, 60]:
            rets = np.array(forward_returns[h])
            if len(rets) > 0:
                pos_prob = float(np.mean(rets > 0))
                neg_prob = float(np.mean(rets < 0))
                statistics[f"{h}d"] = {
                    "mean_return_pct": round(float(np.mean(rets)) * 100.0, 2),
                    "median_return_pct": round(float(np.median(rets)) * 100.0, 2),
                    "positive_prob_pct": round(pos_prob * 100.0, 1),
                    "negative_prob_pct": round(neg_prob * 100.0, 1),
                    "max_gain_pct": round(float(np.max(rets)) * 100.0, 2),
                    "max_loss_pct": round(float(np.min(rets)) * 100.0, 2),
                    "percentiles_pct": {
                        "p10": round(float(np.percentile(rets, 10)) * 100.0, 2),
                        "p25": round(float(np.percentile(rets, 25)) * 100.0, 2),
                        "p50": round(float(np.percentile(rets, 50)) * 100.0, 2),
                        "p75": round(float(np.percentile(rets, 75)) * 100.0, 2),
                        "p90": round(float(np.percentile(rets, 90)) * 100.0, 2)
                    }
                }

        return top_analogues, statistics
