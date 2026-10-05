"""Historical Analogue Search Engine.

Matches today's market state against historical states using normalized feature distance
and calculates forward-return distributions without look-ahead bias.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd


class AnalogueEngine:
    """Historical analogue search and forward outcome statistics engine."""

    def __init__(self, top_k: int = 10):
        self.top_k = top_k
        self.feature_cols = [
            'rsi_14', 'macd_norm', 'volatility_20d', 'distance_sma_50',
            'distance_sma_200', 'return_20d'
        ]

    def find_analogues(
        self,
        features_df: pd.DataFrame,
        current_date: str,
        instrument: str = "GOLD"
    ) -> Dict[str, Any]:
        """
        Finds the top_k most similar historical observation dates before current_date
        and calculates forward return distributions (1D, 3D, 5D, 10D, 20D, 60D).
        """
        inst_df = features_df[features_df['symbol'] == instrument].sort_values('date').copy()
        if inst_df.empty or current_date not in inst_df['date'].values:
            return {"count": 0, "analogues": [], "forward_stats": {}}

        # Normalize features
        if 'macd_norm' not in inst_df.columns and 'macd' in inst_df.columns and 'close' in inst_df.columns:
            inst_df['macd_norm'] = inst_df['macd'] / inst_df['close'].replace(0, np.nan)

        valid_cols = [c for c in self.feature_cols if c in inst_df.columns]
        if not valid_cols:
            return {"count": 0, "analogues": [], "forward_stats": {}}

        current_row = inst_df[inst_df['date'] == current_date].iloc[-1]
        historical_pool = inst_df[inst_df['date'] < current_date].dropna(subset=valid_cols).copy()

        if len(historical_pool) < 5:
            return {"count": 0, "analogues": [], "forward_stats": {}}

        # Calculate Euclidean distance on normalized features
        curr_vec = current_row[valid_cols].values.astype(float)
        hist_matrix = historical_pool[valid_cols].values.astype(float)

        # Standardize using historical statistics
        mean_v = np.nanmean(hist_matrix, axis=0)
        std_v = np.nanstd(hist_matrix, axis=0)
        std_v[std_v == 0] = 1.0

        curr_norm = (curr_vec - mean_v) / std_v
        hist_norm = (hist_matrix - mean_v) / std_v

        dists = np.sqrt(np.nansum((hist_norm - curr_norm) ** 2, axis=1))
        historical_pool['distance'] = dists

        top_analogues = historical_pool.sort_values('distance').head(self.top_k)

        # Calculate forward stats
        horizons = [1, 3, 5, 10, 20, 60]
        stats_by_horizon = {}

        for h in horizons:
            ret_col = f'future_return_{h}d'
            if ret_col in top_analogues.columns:
                returns = top_analogues[ret_col].dropna().values
                if len(returns) > 0:
                    pos_prob = float((returns > 0).mean())
                    stats_by_horizon[f"{h}d"] = {
                        "sample_count": len(returns),
                        "mean_return": float(np.mean(returns)),
                        "median_return": float(np.median(returns)),
                        "positive_probability": round(pos_prob, 4),
                        "negative_probability": round(1.0 - pos_prob, 4),
                        "std_dev": float(np.std(returns)),
                        "max_gain": float(np.max(returns)),
                        "max_loss": float(np.min(returns))
                    }

        analogues_summary = []
        for _, row in top_analogues.iterrows():
            analogues_summary.append({
                "date": row['date'],
                "distance": round(float(row['distance']), 4),
                "close": float(row.get('close', 0.0)),
                "rsi_14": float(row.get('rsi_14', 0.0)),
                "forward_10d_return": float(row.get('future_return_10d', 0.0)) if pd.notna(row.get('future_return_10d')) else None
            })

        return {
            "instrument": instrument,
            "current_date": current_date,
            "sample_count": len(top_analogues),
            "analogues": analogues_summary,
            "forward_stats": stats_by_horizon
        }
