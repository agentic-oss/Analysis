"""
Feature dataset builder for generating ML-ready daily features without look-ahead bias.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any


class FeatureBuilder:
    """Builds comprehensive ML-ready daily feature tables for metals and cross-market data."""

    @staticmethod
    def build_daily_features(
        gold_df: pd.DataFrame,
        silver_df: pd.DataFrame,
        macro_dfs: Dict[str, pd.DataFrame],
        events_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Combines technicals, ratios, macro indicators, and event features into a single daily dataset.
        Future targets are named 'future_gold_ret_Nd' and 'future_silver_ret_Nd'.
        """
        if gold_df.empty:
            return pd.DataFrame()

        # Base gold feature set
        cols_to_keep = [
            "date", "close", "return_1d", "return_5d", "return_20d", "rsi_14", "stoch_rsi",
            "macd", "macd_signal", "macd_histogram", "atr_14", "volatility_20d", "sma_20", "sma_50", "sma_200",
            "dist_sma_20_pct", "dist_sma_50_pct", "dist_sma_200_pct", "dist_52w_high_pct", "dist_52w_low_pct",
            "bollinger_bandwidth"
        ]
        g_cols = [c for c in cols_to_keep if c in gold_df.columns]
        features = gold_df[g_cols].rename(columns={"close": "gold_price", "return_1d": "gold_return_1d"})

        # Merge silver
        if not silver_df.empty and "close" in silver_df.columns:
            s_df = silver_df[["date", "close", "return_1d", "rsi_14", "volatility_20d"]].rename(columns={
                "close": "silver_price",
                "return_1d": "silver_return_1d",
                "rsi_14": "silver_rsi_14",
                "volatility_20d": "silver_volatility_20d"
            })
            features = pd.merge(features, s_df, on="date", how="left")

            # Gold/Silver ratio
            features["gold_silver_ratio"] = features["gold_price"] / features["silver_price"].replace(0, np.nan)
            r_mean = features["gold_silver_ratio"].rolling(252, min_periods=30).mean()
            r_std = features["gold_silver_ratio"].rolling(252, min_periods=30).std()
            features["gold_silver_ratio_zscore"] = (features["gold_silver_ratio"] - r_mean) / r_std.replace(0, np.nan)

        # Merge Macro series
        for m_name, m_df in macro_dfs.items():
            if not m_df.empty and "close" in m_df.columns and "date" in m_df.columns:
                m_sub = m_df[["date", "close"]].rename(columns={"close": f"{m_name}_close"})
                m_sub[f"{m_name}_return_1d"] = m_sub[f"{m_name}_close"].pct_change()
                m_sub[f"{m_name}_return_20d"] = m_sub[f"{m_name}_close"].pct_change(20)
                m_sub[f"{m_name}_change_20d"] = m_sub[f"{m_name}_close"].diff(20)
                features = pd.merge(features, m_sub, on="date", how="left")

        # Event features
        if events_df is not None and not events_df.empty and "date" in events_df.columns:
            features = pd.merge(features, events_df, on="date", how="left")
            features["days_since_event"] = features["days_since_event"].ffill().fillna(999)
            features["days_to_event"] = features["days_to_event"].bfill().fillna(999)
        else:
            features["days_since_event"] = 999
            features["days_to_event"] = 999

        # Generate future targets (isolated from predictors)
        for h in [1, 3, 5, 10, 20, 60]:
            features[f"future_gold_ret_{h}d"] = features["gold_price"].pct_change(h).shift(-h)
            if "silver_price" in features.columns:
                features[f"future_silver_ret_{h}d"] = features["silver_price"].pct_change(h).shift(-h)

        return features.sort_values("date").reset_index(drop=True)
