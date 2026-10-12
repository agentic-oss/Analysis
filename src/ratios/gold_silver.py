import numpy as np
import pandas as pd
from typing import Dict, Any, List

class GoldSilverRatioAnalyzer:
    """Detailed Gold/Silver Ratio and relative value analysis."""

    def __init__(self, lookback: int = 2520): # ~10 years of trading days
        self.lookback = lookback

    def analyze_ratio(
        self,
        gold_df: pd.DataFrame,
        silver_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculates Gold/Silver ratio, historical percentiles, Z-scores,
        and relative performance spreads.
        """
        g_close = gold_df.set_index("date")["close"]
        s_close = silver_df.set_index("date")["close"]

        aligned = pd.concat([g_close, s_close], axis=1, keys=["gold", "silver"]).dropna()
        if aligned.empty:
            return {}

        aligned["ratio"] = aligned["gold"] / aligned["silver"]
        ratio_series = aligned["ratio"]
        current_ratio = float(ratio_series.iloc[-1])

        historical = ratio_series.tail(self.lookback)
        mean_ratio = float(historical.mean())
        std_ratio = float(historical.std())
        min_ratio = float(historical.min())
        max_ratio = float(historical.max())

        z_score = (current_ratio - mean_ratio) / (std_ratio if std_ratio > 0 else 1.0)
        percentile = (historical < current_ratio).mean() * 100.0

        # Relative return spread (Gold return - Silver return)
        g_ret = aligned["gold"].pct_change()
        s_ret = aligned["silver"].pct_change()
        spread_1d = float((g_ret - s_ret).iloc[-1])
        spread_20d = float(((aligned["gold"].pct_change(20)) - (aligned["silver"].pct_change(20))).iloc[-1])

        # Mean reversion research signal
        relative_value_bias = "Neutral"
        if z_score >= 1.5:
            relative_value_bias = "Silver Outperformance Expected (Ratio High)"
        elif z_score <= -1.5:
            relative_value_bias = "Gold Outperformance Expected (Ratio Low)"

        return {
            "current_ratio": round(current_ratio, 2),
            "historical_mean": round(mean_ratio, 2),
            "historical_std": round(std_ratio, 2),
            "min_ratio": round(min_ratio, 2),
            "max_ratio": round(max_ratio, 2),
            "z_score": round(z_score, 2),
            "percentile": round(percentile, 1),
            "spread_1d_pct": round(spread_1d * 100, 2),
            "spread_20d_pct": round(spread_20d * 100, 2),
            "relative_value_bias": relative_value_bias
        }

    def compute_forward_return_spreads(
        self,
        ratio_df: pd.DataFrame,
        extreme_z_threshold: float = 1.5,
        horizons: List[int] = [1, 3, 5, 10, 20, 60]
    ) -> Dict[str, Any]:
        """
        Evaluates historical outcomes following extreme ratio conditions without look-ahead bias.
        """
        df = ratio_df.copy()
        if "ratio" not in df.columns:
            return {}

        mean = df["ratio"].expanding(min_periods=60).mean()
        std = df["ratio"].expanding(min_periods=60).std()
        df["zscore"] = (df["ratio"] - mean) / std.replace(0, np.nan)

        high_ratio_mask = df["zscore"] >= extreme_z_threshold
        low_ratio_mask = df["zscore"] <= -extreme_z_threshold

        results = {"high_ratio_outcomes": {}, "low_ratio_outcomes": {}}

        for h in horizons:
            if "gold" in df.columns and "silver" in df.columns:
                g_fwd = df["gold"].pct_change(h).shift(-h)
                s_fwd = df["silver"].pct_change(h).shift(-h)
                spread_fwd = s_fwd - g_fwd  # Silver return - Gold return

                high_spreads = spread_fwd[high_ratio_mask].dropna()
                low_spreads = spread_fwd[low_ratio_mask].dropna()

                results["high_ratio_outcomes"][f"{h}d"] = {
                    "sample_count": len(high_spreads),
                    "mean_silver_outperformance": round(float(high_spreads.mean() * 100), 2) if len(high_spreads) > 0 else 0.0,
                    "prob_silver_outperforms": round(float((high_spreads > 0).mean() * 100), 1) if len(high_spreads) > 0 else 0.0
                }

                results["low_ratio_outcomes"][f"{h}d"] = {
                    "sample_count": len(low_spreads),
                    "mean_silver_outperformance": round(float(low_spreads.mean() * 100), 2) if len(low_spreads) > 0 else 0.0,
                    "prob_silver_outperforms": round(float((low_spreads > 0).mean() * 100), 1) if len(low_spreads) > 0 else 0.0
                }

        return results
