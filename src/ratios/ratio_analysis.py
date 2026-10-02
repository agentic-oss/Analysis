import numpy as np
import pandas as pd

class GoldSilverRatioAnalyzer:
    """Relative value analyzer for Gold/Silver ratio, spread dynamics, and forward return expectations."""

    def __init__(self, zscore_window: int = 252):
        self.zscore_window = zscore_window

    def analyze_ratio(self, gold_series: pd.Series, silver_series: pd.Series) -> pd.DataFrame:
        """
        Calculates Gold/Silver ratio metrics across time.
        Series should be indexed by date and aligned.
        """
        aligned = pd.concat([gold_series, silver_series], axis=1, keys=["gold", "silver"]).dropna()
        if aligned.empty:
            return pd.DataFrame()

        df = aligned.copy()
        df["ratio"] = df["gold"] / df["silver"]

        # Gold & Silver daily returns
        df["gold_return_1d"] = df["gold"].pct_change()
        df["silver_return_1d"] = df["silver"].pct_change()
        df["spread_return_1d"] = df["gold_return_1d"] - df["silver_return_1d"]

        # Ratio moving average & rolling Z-score
        df["ratio_sma_252"] = df["ratio"].rolling(window=self.zscore_window, min_periods=20).mean()
        df["ratio_std_252"] = df["ratio"].rolling(window=self.zscore_window, min_periods=20).std()
        df["ratio_zscore"] = (df["ratio"] - df["ratio_sma_252"]) / df["ratio_std_252"].replace(0, np.nan)

        # Ratio Percentile rank
        df["ratio_percentile"] = df["ratio"].rolling(window=self.zscore_window, min_periods=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        # Forward return targets (1D, 3D, 5D, 10D, 20D, 60D) for historical research
        horizons = [1, 3, 5, 10, 20, 60]
        for h in horizons:
            df[f"future_gold_return_{h}d"] = df["gold"].pct_change(periods=h).shift(-h)
            df[f"future_silver_return_{h}d"] = df["silver"].pct_change(periods=h).shift(-h)
            df[f"future_ratio_change_{h}d"] = df["ratio"].pct_change(periods=h).shift(-h)

        return df

    def get_latest_summary(self, ratio_df: pd.DataFrame) -> dict:
        """Generates machine-readable latest summary of Gold/Silver relative value status."""
        if ratio_df.empty:
            return {}

        latest = ratio_df.iloc[-1]
        zscore = float(latest.get("ratio_zscore", 0.0) or 0.0)
        percentile = float(latest.get("ratio_percentile", 0.5) or 0.5) * 100.0

        if zscore > 2.0:
            regime = "Ratio Extremely High (Gold Outperforming / Silver Undervalued)"
            bias = "Mean Reversion: Favor Silver over Gold"
        elif zscore < -2.0:
            regime = "Ratio Extremely Low (Silver Outperforming / Gold Undervalued)"
            bias = "Mean Reversion: Favor Gold over Silver"
        elif zscore > 1.0:
            regime = "Ratio Elevated"
            bias = "Moderate Silver Outperformance Preference"
        elif zscore < -1.0:
            regime = "Ratio Depressed"
            bias = "Moderate Gold Outperformance Preference"
        else:
            regime = "Ratio Neutral"
            bias = "Neutral Relative Value"

        return {
            "current_ratio": round(float(latest["ratio"]), 2),
            "ratio_sma_252": round(float(latest.get("ratio_sma_252", latest["ratio"])), 2),
            "ratio_zscore": round(zscore, 2),
            "ratio_percentile": round(percentile, 1),
            "relative_value_regime": regime,
            "research_bias": bias
        }
