import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple


class RelativeValueAnalyzer:
    """Relative value and spread analysis for Gold/Silver ratio and cross-metal pairs."""

    @staticmethod
    def calculate_gold_silver_ratio(
        gold_df: pd.DataFrame,
        silver_df: pd.DataFrame
    ) -> pd.DataFrame:
        if gold_df is None or silver_df is None or gold_df.empty or silver_df.empty:
            return pd.DataFrame()

        g = gold_df[["date", "close"]].rename(columns={"close": "gold_price"}).sort_values("date")
        s = silver_df[["date", "close"]].rename(columns={"close": "silver_price"}).sort_values("date")

        df = pd.merge(g, s, on="date", how="inner")
        if df.empty:
            return pd.DataFrame()

        df["ratio"] = df["gold_price"] / (df["silver_price"] + 1e-10)

        # Returns & spreads
        df["gold_return_1d"] = df["gold_price"].pct_change(1)
        df["silver_return_1d"] = df["silver_price"].pct_change(1)
        df["spread_return_1d"] = df["gold_return_1d"] - df["silver_return_1d"]

        # Long-term statistics
        df["ratio_sma_50"] = df["ratio"].rolling(50, min_periods=10).mean()
        df["ratio_sma_200"] = df["ratio"].rolling(200, min_periods=30).mean()

        mean_252 = df["ratio"].rolling(252, min_periods=30).mean()
        std_252 = df["ratio"].rolling(252, min_periods=30).std()
        df["ratio_zscore_252"] = (df["ratio"] - mean_252) / (std_252 + 1e-10)

        # Percentile rank over rolling 252d and 1000d
        df["ratio_percentile_252"] = df["ratio"].rolling(252, min_periods=30).apply(
            lambda x: (pd.Series(x).rank().iloc[-1] / len(x)) * 100.0, raw=False
        )

        # Mean reversion indicator (distance from 200 SMA in Z-score units)
        df["mean_reversion_signal"] = np.where(df["ratio_zscore_252"] > 2.0, "EXTREME_HIGH_MEAN_REVERT",
                                      np.where(df["ratio_zscore_252"] < -2.0, "EXTREME_LOW_MEAN_REVERT", "NEUTRAL"))

        return df

    @staticmethod
    def analyze_ratio_extremes_and_forward_outcomes(
        ratio_df: pd.DataFrame,
        zscore_threshold: float = 1.8
    ) -> Dict[str, Any]:
        if ratio_df is None or ratio_df.empty or len(ratio_df) < 60:
            return {}

        df = ratio_df.copy()

        # Forward return columns
        for h in [1, 3, 5, 10, 20, 60]:
            df[f"fwd_ratio_change_{h}d"] = (df["ratio"].shift(-h) - df["ratio"]) / df["ratio"]
            df[f"fwd_gold_ret_{h}d"] = df["gold_price"].pct_change(h).shift(-h)
            df[f"fwd_silver_ret_{h}d"] = df["silver_price"].pct_change(h).shift(-h)

        high_extremes = df[df["ratio_zscore_252"] >= zscore_threshold]
        low_extremes = df[df["ratio_zscore_252"] <= -zscore_threshold]

        def summarize_outcomes(sub_df: pd.DataFrame) -> Dict[str, Any]:
            if sub_df.empty:
                return {"sample_count": 0}
            res = {"sample_count": len(sub_df)}
            for h in [1, 3, 5, 10, 20, 60]:
                col_g = f"fwd_gold_ret_{h}d"
                col_s = f"fwd_silver_ret_{h}d"
                col_r = f"fwd_ratio_change_{h}d"

                g_vals = sub_df[col_g].dropna()
                s_vals = sub_df[col_s].dropna()
                r_vals = sub_df[col_r].dropna()

                res[f"{h}d"] = {
                    "gold_mean_return": round(float(g_vals.mean()), 4) if len(g_vals) > 0 else 0.0,
                    "silver_mean_return": round(float(s_vals.mean()), 4) if len(s_vals) > 0 else 0.0,
                    "ratio_mean_change": round(float(r_vals.mean()), 4) if len(r_vals) > 0 else 0.0,
                    "silver_outperform_prob": round(float((s_vals > g_vals).mean()), 4) if len(s_vals) > 0 else 0.0
                }
            return res

        latest = df.iloc[-1]
        summary = {
            "latest_ratio": round(float(latest["ratio"]), 2),
            "latest_zscore": round(float(latest["ratio_zscore_252"]), 2) if not pd.isna(latest["ratio_zscore_252"]) else 0.0,
            "latest_percentile": round(float(latest["ratio_percentile_252"]), 1) if not pd.isna(latest["ratio_percentile_252"]) else 50.0,
            "mean_reversion_signal": str(latest["mean_reversion_signal"]),
            "historical_high_extremes_outcomes": summarize_outcomes(high_extremes),
            "historical_low_extremes_outcomes": summarize_outcomes(low_extremes)
        }

        return summary
