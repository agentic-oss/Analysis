import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

class FeatureStoreBuilder:
    """Builds ML-ready daily feature store datasets with future target variables without look-ahead bias."""

    @staticmethod
    def build_daily_features(
        processed_dict: Dict[str, pd.DataFrame],
        ratio_df: pd.DataFrame,
        target_symbol: str = "GOLD",
        horizons: List[int] = [1, 3, 5, 10, 20, 60]
    ) -> pd.DataFrame:
        """Combine technical indicators, macro variables, ratios, and future return targets into a unified dataset for a specific target instrument."""
        target_df = processed_dict.get(target_symbol)

        if target_df is None or target_df.empty:
            return pd.DataFrame()

        # Start with Target instrument as main timeline
        df = target_df[["date", "symbol", "close", "open", "high", "low", "volume"]].copy()
        df = df.rename(columns={"close": "price", "symbol": "instrument"})

        # Past returns (No look-ahead)
        df["return_1d"] = df["price"].pct_change(1)
        df["return_5d"] = df["price"].pct_change(5)
        df["return_20d"] = df["price"].pct_change(20)
        df["return_60d"] = df["price"].pct_change(60)

        # Technical indicators from target_df
        tech_cols = [c for c in target_df.columns if c not in ["date", "symbol", "ticker", "open", "high", "low", "close", "volume"]]
        for col in tech_cols:
            df[col] = target_df[col]

        # Merge Gold/Silver Ratio features
        if ratio_df is not None and not ratio_df.empty:
            ratio_sub = ratio_df[["date", "gold_silver_ratio", "ratio_zscore", "ratio_percentile_252"]].copy()
            df = pd.merge(df, ratio_sub, on="date", how="left")
            df["gold_silver_ratio_zscore"] = df["ratio_zscore"]

        # Merge Macro Features
        macro_keys = [("DXY", "dxy"), ("US10Y", "us10y"), ("CRUDE_OIL", "oil"), ("SP500", "sp500"), ("NIFTY50", "nifty"), ("VIX", "vix")]
        for key, prefix in macro_keys:
            m_df = processed_dict.get(key)
            if m_df is not None and not m_df.empty:
                m_sub = m_df[["date", "close"]].rename(columns={"close": f"{prefix}_close"})
                df = pd.merge(df, m_sub, on="date", how="left")
                df[f"{prefix}_return_1d"] = df[f"{prefix}_close"].pct_change(1)
                df[f"{prefix}_return_20d"] = df[f"{prefix}_close"].pct_change(20)
                if key == "US10Y":
                    df["us10y_change_20d"] = df["us10y_close"].diff(20)
                if key == "VIX":
                    df["vix_level"] = df["vix_close"]

        # Future Targets (LOOK-AHEAD PROTECTED - Shifted backwards in time for training targets)
        for h in horizons:
            # target on date T = (Price at T+h - Price at T) / Price at T
            df[f"future_return_{h}d"] = (df["price"].shift(-h) - df["price"]) / df["price"]

        return df

    @staticmethod
    def add_event_features(
        feature_df: pd.DataFrame,
        events_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add event features like days_to_event and days_since_event."""
        if feature_df.empty or events_df is None or events_df.empty:
            feature_df["days_to_event"] = 999
            feature_df["days_since_event"] = 999
            feature_df["event_type"] = "none"
            return feature_df

        df = feature_df.copy()
        df["date_dt"] = pd.to_datetime(df["date"])
        events_df["event_date_dt"] = pd.to_datetime(events_df["event_date"])

        days_to = []
        days_since = []

        for curr_date in df["date_dt"]:
            future_events = events_df[events_df["event_date_dt"] >= curr_date]
            past_events = events_df[events_df["event_date_dt"] <= curr_date]

            d_to = (future_events["event_date_dt"].min() - curr_date).days if not future_events.empty else 999
            d_since = (curr_date - past_events["event_date_dt"].max()).days if not past_events.empty else 999

            days_to.append(d_to if pd.notnull(d_to) else 999)
            days_since.append(d_since if pd.notnull(d_since) else 999)

        df["days_to_event"] = days_to
        df["days_since_event"] = days_since
        df = df.drop(columns=["date_dt"])
        return df
