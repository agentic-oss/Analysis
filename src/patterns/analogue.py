import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

class AnalogueEngine:
    """Finds historical analog dates matching today's feature vector without look-ahead bias."""

    def __init__(self, top_k: int = 25):
        self.top_k = top_k

    def find_analogues(
        self,
        target_features: pd.Series,
        historical_features_df: pd.DataFrame,
        feature_cols: List[str],
        horizons: List[int] = [1, 3, 5, 10, 20, 60]
    ) -> Dict[str, Any]:
        """Calculates distance between target date features and all historical dates, returning top_k matches and forward returns."""
        if historical_features_df.empty or len(historical_features_df) < 10:
            return {"sample_count": 0, "analogues": [], "forward_stats": {}}

        # Filter available feature columns
        cols = [c for c in feature_cols if c in historical_features_df.columns and c in target_features.index]
        if not cols:
            return {"sample_count": 0, "analogues": [], "forward_stats": {}}

        hist = historical_features_df.copy().dropna(subset=cols)
        if hist.empty:
            return {"sample_count": 0, "analogues": [], "forward_stats": {}}

        # Compute Euclidean distance on standardized features
        t_vals = target_features[cols].astype(float).values
        h_vals = hist[cols].astype(float).values

        mean = np.nanmean(h_vals, axis=0)
        std = np.nanstd(h_vals, axis=0)
        std[std == 0] = 1.0

        norm_t = (t_vals - mean) / std
        norm_h = (h_vals - mean) / std

        distances = np.linalg.norm(norm_h - norm_t, axis=1)
        hist["distance"] = distances

        # Get top k closest matches (excluding exact target date if present)
        top_matches = hist.sort_values("distance").head(self.top_k)

        # Calculate forward statistics for each horizon
        stats_by_horizon = {}
        for h in horizons:
            col = f"future_return_{h}d"
            if col in top_matches.columns:
                returns = top_matches[col].dropna()
                if not returns.empty:
                    pos_prob = float((returns > 0).mean())
                    stats_by_horizon[f"{h}d"] = {
                        "horizon_days": h,
                        "sample_count": len(returns),
                        "mean_return": float(returns.mean()),
                        "median_return": float(returns.median()),
                        "std_dev": float(returns.std()) if len(returns) > 1 else 0.0,
                        "positive_probability": pos_prob,
                        "negative_probability": 1.0 - pos_prob,
                        "max_gain": float(returns.max()),
                        "max_loss": float(returns.min()),
                        "p10": float(np.percentile(returns, 10)),
                        "p90": float(np.percentile(returns, 90))
                    }

        analogue_list = []
        for idx, row in top_matches.iterrows():
            analogue_list.append({
                "date": str(row.get("date", idx)),
                "distance": float(row["distance"])
            })

        return {
            "sample_count": len(top_matches),
            "analogues": analogue_list,
            "forward_stats": stats_by_horizon
        }


class ForwardProbabilityEngine:
    """Transforms historical analogue distributions and features into conditional probability forecasts."""

    @staticmethod
    def generate_probabilistic_signals(
        analogue_stats: Dict[str, Any],
        disclaimer: str = "Historical conditional statistics only, not guaranteed future outcomes."
    ) -> Dict[str, Any]:
        signals = {}
        forward_stats = analogue_stats.get("forward_stats", {})

        for horizon, stat in forward_stats.items():
            pos_prob = stat.get("positive_probability", 0.5)
            exp_ret = stat.get("mean_return", 0.0)

            bias = "Neutral"
            if pos_prob >= 0.60 and exp_ret > 0.005:
                bias = "Bullish"
            elif pos_prob <= 0.40 and exp_ret < -0.005:
                bias = "Bearish"

            signals[horizon] = {
                "horizon": horizon,
                "bias": bias,
                "positive_return_probability": pos_prob,
                "negative_return_probability": stat.get("negative_probability", 0.5),
                "expected_return": exp_ret,
                "median_return": stat.get("median_return", 0.0),
                "expected_volatility": stat.get("std_dev", 0.0),
                "max_gain": stat.get("max_gain", 0.0),
                "max_loss": stat.get("max_loss", 0.0)
            }

        return {
            "disclaimer": disclaimer,
            "signals": signals
        }
