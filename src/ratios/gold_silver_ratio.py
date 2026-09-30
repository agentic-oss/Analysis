import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GoldSilverRatioAnalysis:
    """Analyzes the Gold/Silver ratio (gold_price / silver_price), percentile, Z-score, and mean reversion behavior."""

    @staticmethod
    def calculate_ratio_metrics(
        gold_df: pd.DataFrame, silver_df: pd.DataFrame, rolling_window: int = 252
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Calculates:
        - Ratio: gold_price / silver_price
        - Historical percentile
        - Long-term mean & Z-score
        - Forward return conditional statistics
        """
        if gold_df.empty or silver_df.empty:
            return pd.DataFrame(), {}

        merged = pd.merge(
            gold_df[["date", "close"]].rename(columns={"close": "gold_price"}),
            silver_df[["date", "close"]].rename(columns={"close": "silver_price"}),
            on="date",
            how="inner",
        ).sort_values("date")

        if merged.empty or (merged["silver_price"] <= 0).any():
            return pd.DataFrame(), {}

        df = merged.copy()
        df["ratio"] = df["gold_price"] / df["silver_price"]

        # Moving statistics
        df["ratio_sma_200"] = df["ratio"].rolling(window=200).mean()
        df["ratio_mean"] = df["ratio"].expanding(min_periods=50).mean()
        df["ratio_std"] = df["ratio"].expanding(min_periods=50).std()
        df["ratio_zscore"] = (df["ratio"] - df["ratio_mean"]) / df["ratio_std"].replace(0, np.nan)

        # Percentile calculation
        df["ratio_percentile"] = df["ratio"].expanding(min_periods=50).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100.0, raw=False
        )

        latest = df.iloc[-1]
        summary = {
            "current_ratio": float(latest["ratio"]),
            "ratio_sma_200": float(latest["ratio_sma_200"]) if pd.notnull(latest["ratio_sma_200"]) else None,
            "z_score": float(latest["ratio_zscore"]) if pd.notnull(latest["ratio_zscore"]) else None,
            "percentile": float(latest["ratio_percentile"]) if pd.notnull(latest["ratio_percentile"]) else None,
            "is_extreme_high": bool(latest["ratio_zscore"] > 2.0) if pd.notnull(latest["ratio_zscore"]) else False,
            "is_extreme_low": bool(latest["ratio_zscore"] < -2.0) if pd.notnull(latest["ratio_zscore"]) else False,
        }

        return df, summary

    @staticmethod
    def calculate_forward_spread_returns(
        ratio_df: pd.DataFrame, horizons: list = [1, 3, 5, 10, 20, 60]
    ) -> pd.DataFrame:
        """Calculates forward ratio changes and relative gold/silver performance for research signals."""
        if ratio_df.empty or "ratio" not in ratio_df.columns:
            return ratio_df

        df = ratio_df.copy()
        for h in horizons:
            # Future change in ratio over h days
            df[f"future_ratio_change_{h}d"] = (df["ratio"].shift(-h) - df["ratio"]) / df["ratio"] * 100.0

        return df
