import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

class HistoricalAnalogueEngine:
    """Finds historical market conditions similar to current state and computes forward return statistics without look-ahead bias."""

    def __init__(self, feature_cols: list[str] = None, top_k: int = 15):
        self.feature_cols = feature_cols or [
            "rsi_14", "volatility_20d", "dist_sma_200", "dist_sma_20",
            "gold_silver_ratio_zscore", "dxy_return_20d", "us10y_change_20d", "vix_level"
        ]
        self.top_k = top_k

    def find_analogues(self, current_features: pd.Series, historical_df: pd.DataFrame) -> dict:
        """
        Searches historical database for closest matching days using standardized distance.
        current_features: pandas Series of feature values for date T.
        historical_df: DataFrame of historical days (strictly before T) with features and future returns.
        """
        if historical_df.empty:
            return {}

        # Filter available feature columns
        available_cols = [c for c in self.feature_cols if c in historical_df.columns and c in current_features.index]
        if not available_cols:
            return {}

        # Fill missing values in historical features with median
        hist_feats = historical_df[available_cols].copy().apply(pd.to_numeric, errors="coerce")
        hist_feats = hist_feats.fillna(hist_feats.median()).fillna(0.0)

        if hist_feats.empty:
            return {}

        curr_series = pd.to_numeric(current_features[available_cols], errors="coerce").fillna(hist_feats.median()).fillna(0.0)
        curr_vector = curr_series.values.astype(float).reshape(1, -1)

        # Standardize using historical distribution only
        mean = hist_feats.mean(axis=0).astype(float)
        std = hist_feats.std(axis=0).replace(0, 1.0).astype(float)

        hist_scaled = ((hist_feats - mean) / std).astype(float).values
        curr_scaled = ((curr_vector - mean.values) / std.values).astype(float)

        # Calculate Euclidean distances
        distances = cdist(curr_scaled, hist_scaled, metric="euclidean").flatten()

        hist_subset = historical_df.loc[hist_feats.index].copy()
        hist_subset["distance"] = distances
        hist_subset["similarity_score"] = 1.0 / (1.0 + distances)

        # Sort by distance
        top_matches = hist_subset.sort_values(by="distance").head(self.top_k)

        # Calculate forward statistics for each horizon
        horizons = [1, 3, 5, 10, 20, 60]
        stats = {}

        for h in horizons:
            col = f"future_return_{h}d"
            if col in top_matches.columns:
                returns = top_matches[col].dropna()
                if not returns.empty:
                    pos_prob = (returns > 0).mean()
                    neg_prob = (returns < 0).mean()
                    stats[f"{h}d"] = {
                        "sample_count": len(returns),
                        "mean_return_pct": round(float(returns.mean() * 100), 2),
                        "median_return_pct": round(float(returns.median() * 100), 2),
                        "win_rate_pct": round(float(pos_prob * 100), 1),
                        "positive_probability": round(float(pos_prob), 2),
                        "negative_probability": round(float(neg_prob), 2),
                        "min_return_pct": round(float(returns.min() * 100), 2),
                        "max_return_pct": round(float(returns.max() * 100), 2),
                        "p25_pct": round(float(returns.quantile(0.25) * 100), 2),
                        "p75_pct": round(float(returns.quantile(0.75) * 100), 2),
                    }

        analogue_dates = top_matches["date"].astype(str).tolist() if "date" in top_matches.columns else []

        return {
            "num_analogues": len(top_matches),
            "analogue_dates": analogue_dates,
            "top_similarity_scores": [round(s, 3) for s in top_matches["similarity_score"].tolist()],
            "forward_statistics": stats,
            "disclaimer": "Historical conditional statistics are probabilistic research signals, not guaranteed future prices."
        }
