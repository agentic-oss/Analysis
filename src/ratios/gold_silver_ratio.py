import pandas as pd
import numpy as np
from typing import Dict, Any, List


class GoldSilverRatioAnalyzer:
    """Performs Gold/Silver ratio analysis and relative-value metrics."""

    @staticmethod
    def calculate_ratio_series(
        gold_df: pd.DataFrame, silver_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge Gold and Silver data and compute Gold/Silver ratio metrics."""
        if gold_df is None or silver_df is None or gold_df.empty or silver_df.empty:
            return pd.DataFrame()

        g = gold_df[["Date", "Close"]].copy().rename(columns={"Close": "gold_close"})
        s = silver_df[["Date", "Close"]].copy().rename(columns={"Close": "silver_close"})

        g["Date"] = pd.to_datetime(g["Date"])
        s["Date"] = pd.to_datetime(s["Date"])

        df = pd.merge(g, s, on="Date", how="inner").sort_values("Date").reset_index(drop=True)
        df["ratio"] = df["gold_close"] / (df["silver_close"] + 1e-9)

        # Returns
        df["gold_return"] = df["gold_close"].pct_change()
        df["silver_return"] = df["silver_close"].pct_change()
        df["return_spread"] = df["gold_return"] - df["silver_return"]

        # Ratio rolling stats
        df["ratio_sma_50"] = df["ratio"].rolling(window=50).mean()
        df["ratio_sma_200"] = df["ratio"].rolling(window=200).mean()

        ratio_mean = df["ratio"].rolling(window=252, min_periods=20).mean()
        ratio_std = df["ratio"].rolling(window=252, min_periods=20).std()
        df["ratio_zscore"] = (df["ratio"] - ratio_mean) / (ratio_std + 1e-9)

        # Percentile ranking (rolling 252 days)
        def calc_percentile(s):
            if len(s) == 0:
                return 50.0
            val = s.iloc[-1]
            return (s < val).mean() * 100.0

        df["ratio_percentile"] = df["ratio"].rolling(window=252, min_periods=20).apply(calc_percentile, raw=False)

        return df

    @staticmethod
    def analyze_latest_ratio(df: pd.DataFrame) -> Dict[str, Any]:
        """Generate high-level metrics for the latest ratio state."""
        if df is None or df.empty or "ratio" not in df.columns:
            return {}

        latest = df.iloc[-1]
        ratio_val = float(latest["ratio"])
        zscore = float(latest.get("ratio_zscore", 0.0))
        percentile = float(latest.get("ratio_percentile", 50.0))

        long_term_avg = float(df["ratio"].mean())

        # Regime classification for GSR
        if zscore > 2.0:
            regime = "Gold Extreme Outperformance / Silver Historically Cheap"
        elif zscore < -2.0:
            regime = "Silver Extreme Outperformance / Gold Historically Cheap"
        elif zscore > 1.0:
            regime = "Elevated Ratio (Favors Silver Mean-Reversion)"
        elif zscore < -1.0:
            regime = "Depressed Ratio (Favors Gold Mean-Reversion)"
        else:
            regime = "Normal Ratio Range"

        return {
            "current_ratio": round(ratio_val, 2),
            "zscore": round(zscore, 2),
            "percentile": round(percentile, 1),
            "long_term_average": round(long_term_avg, 2),
            "regime": regime,
            "sma_50": round(float(latest.get("ratio_sma_50", ratio_val)), 2),
            "sma_200": round(float(latest.get("ratio_sma_200", ratio_val)), 2),
        }

    @staticmethod
    def calculate_historical_extreme_forward_returns(
        df: pd.DataFrame, z_threshold: float = 2.0
    ) -> Dict[str, Any]:
        """
        Identify historical extreme ratio conditions (Z-score > threshold or < -threshold)
        and calculate subsequent forward returns for Gold and Silver.
        """
        if df is None or df.empty or "ratio_zscore" not in df.columns:
            return {}

        df = df.copy()
        horizons = [1, 3, 5, 10, 20, 60]
        for h in horizons:
            df[f"gold_fwd_{h}d"] = df["gold_close"].pct_change(h).shift(-h)
            df[f"silver_fwd_{h}d"] = df["silver_close"].pct_change(h).shift(-h)

        high_extreme = df[df["ratio_zscore"] > z_threshold]
        low_extreme = df[df["ratio_zscore"] < -z_threshold]

        res = {"high_ratio_extremes": {}, "low_ratio_extremes": {}}

        for name, sub in [("high_ratio_extremes", high_extreme), ("low_ratio_extremes", low_extreme)]:
            res[name]["sample_count"] = len(sub)
            for h in horizons:
                g_ret = sub[f"gold_fwd_{h}d"].dropna()
                s_ret = sub[f"silver_fwd_{h}d"].dropna()
                res[name][f"{h}d"] = {
                    "gold_mean_return_pct": round(float(g_ret.mean() * 100), 2) if len(g_ret) > 0 else 0.0,
                    "silver_mean_return_pct": round(float(s_ret.mean() * 100), 2) if len(s_ret) > 0 else 0.0,
                    "gold_win_rate": round(float((g_ret > 0).mean() * 100), 1) if len(g_ret) > 0 else 0.0,
                    "silver_win_rate": round(float((s_ret > 0).mean() * 100), 1) if len(s_ret) > 0 else 0.0,
                }

        return res
