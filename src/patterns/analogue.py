"""Historical Analogue Engine for multi-factor market state similarity matching."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List


class HistoricalAnalogueEngine:
    """
    Searches historical feature store for dates with similar multi-factor states (RSI, MACD, MAs, Volatility, Ratio, DXY, Rates)
    without look-ahead bias, computing forward return distributions across horizons (1D, 3D, 5D, 10D, 20D, 60D).
    """

    def __init__(self, top_k: int = 15, min_similarity: float = 0.50):
        self.top_k = top_k
        self.min_similarity = min_similarity

    def find_analogues(self,
                        feature_df: pd.DataFrame,
                        target_date: str,
                        horizons: List[int] = [1, 3, 5, 10, 20, 60]) -> Dict[str, Any]:
        """
        Finds top K historical dates similar to target_date's feature vector.
        Ensures strict temporal safety (only historical dates strictly < target_date).
        """
        if feature_df.empty or 'date' not in feature_df.columns or target_date not in feature_df['date'].values:
            return {"top_analogues": [], "forward_stats": {}}

        df = feature_df.sort_values('date').reset_index(drop=True)
        target_idx = df[df['date'] == target_date].index[0]

        # Strictly historical observation pool (< target_idx - max(horizons) to allow complete forward returns evaluation)
        max_horizon = max(horizons)
        pool_df = df.iloc[:target_idx].copy()
        pool_df = pool_df[pool_df.index <= (len(df) - max_horizon - 1)].copy()

        if len(pool_df) < 10:
            return {"top_analogues": [], "forward_stats": {}}

        # Feature vector columns for cosine/Euclidean similarity
        feature_cols = [c for c in [
            'rsi_14', 'macd', 'stoch_rsi', 'atr_14', 'volatility_20d',
            'dist_sma_20_pct', 'dist_sma_50_pct', 'dist_sma_200_pct',
            'gold_silver_ratio', 'dxy_return_20d', 'us10y_change_20d'
        ] if c in df.columns]

        if not feature_cols:
            return {"top_analogues": [], "forward_stats": {}}

        target_vec = df.loc[target_idx, feature_cols].fillna(0).values.astype(float)
        pool_vecs = pool_df[feature_cols].fillna(0).values.astype(float)

        # Normalize features (z-score scaling based on pool)
        mean_vec = np.mean(pool_vecs, axis=0)
        std_vec = np.std(pool_vecs, axis=0) + 1e-8

        target_norm = (target_vec - mean_vec) / std_vec
        pool_norm = (pool_vecs - mean_vec) / std_vec

        # Distance & Similarity score
        distances = np.linalg.norm(pool_norm - target_norm, axis=1)
        similarities = 1.0 / (1.0 + distances)

        pool_df['similarity_score'] = similarities
        top_matches = pool_df[pool_df['similarity_score'] >= self.min_similarity].sort_values('similarity_score', ascending=False).head(self.top_k)

        analogues_list = []
        forward_returns_by_horizon: Dict[int, List[float]] = {h: [] for h in horizons}

        for _, row in top_matches.iterrows():
            match_idx = row.name
            match_date = str(row['date'])
            match_sim = float(row['similarity_score'])

            match_info = {
                "date": match_date,
                "similarity_score": round(match_sim, 4),
                "forward_returns_pct": {}
            }

            for h in horizons:
                fut_idx = match_idx + h
                if fut_idx < len(df):
                    p_curr = float(df.loc[match_idx, 'close'])
                    p_fut = float(df.loc[fut_idx, 'close'])
                    ret = (p_fut - p_curr) / p_curr if p_curr else 0.0
                    match_info["forward_returns_pct"][f"{h}d"] = round(ret * 100, 2)
                    forward_returns_by_horizon[h].append(ret)

            analogues_list.append(match_info)

        # Aggregate forward return statistics across horizons
        forward_stats = {}
        for h in horizons:
            rets = np.array(forward_returns_by_horizon[h])
            if len(rets) > 0:
                pos_prob = float((rets > 0).mean())
                neg_prob = float((rets < 0).mean())
                forward_stats[f"{h}d"] = {
                    "sample_count": len(rets),
                    "mean_return_pct": round(float(rets.mean() * 100), 2),
                    "median_return_pct": round(float(np.median(rets) * 100), 2),
                    "positive_prob_pct": round(pos_prob * 100, 1),
                    "negative_prob_pct": round(neg_prob * 100, 1),
                    "max_gain_pct": round(float(rets.max() * 100), 2),
                    "max_loss_pct": round(float(rets.min() * 100), 2)
                }
            else:
                forward_stats[f"{h}d"] = {
                    "sample_count": 0,
                    "mean_return_pct": 0.0,
                    "median_return_pct": 0.0,
                    "positive_prob_pct": 50.0,
                    "negative_prob_pct": 50.0,
                    "max_gain_pct": 0.0,
                    "max_loss_pct": 0.0
                }

        return {
            "target_date": target_date,
            "top_analogues": analogues_list,
            "forward_stats": forward_stats
        }
