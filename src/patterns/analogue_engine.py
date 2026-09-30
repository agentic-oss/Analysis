import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HistoricalAnalogueEngine:
    """
    Finds historical market situations similar to current market state using multi-factor distance scoring,
    and computes forward return distributions (1D, 3D, 5D, 10D, 20D, 60D) without look-ahead bias.
    """

    FEATURE_WEIGHTS = {
        "rsi_14": 1.0,
        "macd_norm": 1.0,
        "dist_sma_20": 1.0,
        "dist_sma_200": 1.5,
        "volatility_20d": 1.2,
        "gold_silver_ratio_z": 1.5,
        "dxy_return_20d": 1.0,
        "us10y_change_20d": 1.0,
        "vix_level": 0.8,
    }

    def __init__(self, top_k: int = 25, min_lookback: int = 252):
        self.top_k = top_k
        self.min_lookback = min_lookback

    def find_analogues(
        self, historical_df: pd.DataFrame, current_state: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Calculates similarity scoring against historical rows.
        Excludes recent history (last 60 trading days) to ensure forward targets exist.
        """
        if historical_df.empty or len(historical_df) < self.min_lookback:
            return {"count": 0, "analogues": [], "forward_stats": {}}

        df = historical_df.copy().sort_values("date").reset_index(drop=True)

        # Retain only historical rows with complete forward targets (exclude last 60 days)
        eval_df = df.iloc[:-60].copy() if len(df) > 120 else df.copy()

        if eval_df.empty:
            return {"count": 0, "analogues": [], "forward_stats": {}}

        # Calculate weighted Euclidean distance across normalized features
        distance_series = pd.Series(0.0, index=eval_df.index)
        weight_sum = 0.0

        for feat, weight in self.FEATURE_WEIGHTS.items():
            if feat in eval_df.columns and feat in current_state and current_state[feat] is not None:
                # Z-score normalize feature in history
                std = eval_df[feat].std()
                mean = eval_df[feat].mean()
                if std > 0:
                    hist_norm = (eval_df[feat] - mean) / std
                    curr_norm = (current_state[feat] - mean) / std
                    distance_series += weight * ((hist_norm - curr_norm) ** 2)
                    weight_sum += weight

        if weight_sum == 0:
            return {"count": 0, "analogues": [], "forward_stats": {}}

        distance_series = np.sqrt(distance_series / weight_sum)
        eval_df["analogue_distance"] = distance_series

        # Select top-k smallest distances
        top_analogues = eval_df.sort_values("analogue_distance").head(self.top_k)

        # Calculate forward statistics across horizons
        forward_stats = {}
        for h in [1, 3, 5, 10, 20, 60]:
            col = f"future_return_{h}d"
            if col in top_analogues.columns:
                returns = top_analogues[col].dropna()
                if not returns.empty:
                    pos_prob = float((returns > 0).mean())
                    forward_stats[f"{h}d"] = {
                        "sample_size": len(returns),
                        "mean_return_pct": float(returns.mean() * 100.0),
                        "median_return_pct": float(returns.median() * 100.0),
                        "std_pct": float(returns.std() * 100.0),
                        "win_probability": float(pos_prob * 100.0),
                        "max_gain_pct": float(returns.max() * 100.0),
                        "max_loss_pct": float(returns.min() * 100.0),
                        "p25_pct": float(returns.quantile(0.25) * 100.0),
                        "p75_pct": float(returns.quantile(0.75) * 100.0),
                    }

        analogue_list = []
        for _, row in top_analogues.iterrows():
            analogue_list.append(
                {
                    "date": str(row["date"]),
                    "distance": float(row["analogue_distance"]),
                    "forward_10d_return_pct": float(row.get("future_return_10d", 0.0) * 100.0),
                    "forward_20d_return_pct": float(row.get("future_return_20d", 0.0) * 100.0),
                }
            )

        return {
            "top_k": len(top_analogues),
            "analogues": analogue_list,
            "forward_stats": forward_stats,
        }
