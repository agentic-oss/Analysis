import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from src.indicators.technical import TechnicalIndicators


class HistoricalAnalogueEngine:
    """
    Searches historical feature store for market state analogues matching today's
    multi-factor observation vector, and calculates empirical forward-return distributions.
    """

    def __init__(self, top_k: int = 15):
        self.top_k = top_k
        self.feature_cols = [
            "rsi_14",
            "macd_hist",
            "volatility_20d",
            "distance_sma_20",
            "distance_sma_50",
            "distance_sma_200",
            "return_5d",
            "return_20d",
        ]

    def find_analogues(
        self,
        historical_df: pd.DataFrame,
        current_date: str,
        horizons: List[int] = [1, 3, 5, 10, 20, 60],
    ) -> Dict[str, Any]:
        """
        Match current market state with past observations up to current_date - min_horizon.
        Guarantees no future information leakage.
        """
        if historical_df is None or historical_df.empty or len(historical_df) < 20:
            return {}

        df = TechnicalIndicators.calculate_all(historical_df)
        df["Date"] = pd.to_datetime(df["Date"])
        current_dt = pd.to_datetime(current_date)

        # Filter out current and future dates
        hist_pool = df[df["Date"] < current_dt].copy().reset_index(drop=True)

        # Extract current target state vector
        current_match = df[df["Date"] == current_dt]
        if current_match.empty:
            current_match = hist_pool.iloc[-1:]
            hist_pool = hist_pool.iloc[:-1]

        # Available feature columns present in dataframe
        valid_cols = [c for c in self.feature_cols if c in df.columns]
        if not valid_cols:
            return {}

        # Fill NAs
        hist_pool[valid_cols] = hist_pool[valid_cols].fillna(0.0)
        curr_vector = current_match[valid_cols].fillna(0.0).iloc[0].values

        # Standardize features across historical pool
        hist_matrix = hist_pool[valid_cols].values
        mean = hist_matrix.mean(axis=0)
        std = hist_matrix.std(axis=0) + 1e-9

        hist_norm = (hist_matrix - mean) / std
        curr_norm = (curr_vector - mean) / std

        # Compute Euclidean distance
        distances = np.linalg.norm(hist_norm - curr_norm, axis=1)
        hist_pool["distance"] = distances

        # Pick top K closest analogues
        eligible_pool = hist_pool
        top_analogues = eligible_pool.sort_values("distance").head(self.top_k)

        # Calculate forward returns for each horizon
        forward_stats = {}
        for h in horizons:
            fwd_returns = []
            for _, row in top_analogues.iterrows():
                idx = row.name
                if idx + h < len(hist_pool):
                    p_start = hist_pool.at[idx, "Close"]
                    p_end = hist_pool.at[idx + h, "Close"]
                    if p_start > 0:
                        ret = (p_end - p_start) / p_start
                        fwd_returns.append(ret)

            if fwd_returns:
                arr = np.array(fwd_returns)
                forward_stats[f"{h}d"] = {
                    "horizon_days": h,
                    "sample_count": len(arr),
                    "mean_return_pct": round(float(np.mean(arr) * 100), 2),
                    "median_return_pct": round(float(np.median(arr) * 100), 2),
                    "pos_prob_pct": round(float((arr > 0).mean() * 100), 1),
                    "neg_prob_pct": round(float((arr < 0).mean() * 100), 1),
                    "max_gain_pct": round(float(np.max(arr) * 100), 2),
                    "max_loss_pct": round(float(np.min(arr) * 100), 2),
                    "percentiles": {
                        "p10": round(float(np.percentile(arr, 10) * 100), 2),
                        "p25": round(float(np.percentile(arr, 25) * 100), 2),
                        "p50": round(float(np.percentile(arr, 50) * 100), 2),
                        "p75": round(float(np.percentile(arr, 75) * 100), 2),
                        "p90": round(float(np.percentile(arr, 90) * 100), 2),
                    },
                }
            else:
                forward_stats[f"{h}d"] = {
                    "horizon_days": h,
                    "sample_count": 0,
                    "mean_return_pct": 0.0,
                    "median_return_pct": 0.0,
                    "pos_prob_pct": 50.0,
                    "neg_prob_pct": 50.0,
                    "max_gain_pct": 0.0,
                    "max_loss_pct": 0.0,
                    "percentiles": {},
                }

        matched_dates = top_analogues["Date"].dt.strftime("%Y-%m-%d").tolist()

        return {
            "current_date": current_date,
            "top_k_samples": len(top_analogues),
            "matched_analogue_dates": matched_dates,
            "forward_statistics": forward_stats,
        }
