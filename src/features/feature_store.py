"""
ML Feature Store Engine.
Generates daily feature datasets with target variables and event features.
Ensures zero look-ahead bias by maintaining strict temporal isolation.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class FeatureStoreEngine:
    """Builds ML-ready daily feature store and future target variables."""

    def __init__(self, forecast_horizons: List[int] = [1, 3, 5, 10, 20, 60]):
        self.forecast_horizons = forecast_horizons

    def build_daily_features(
        self,
        metals_df: pd.DataFrame,
        macro_df: Optional[pd.DataFrame] = None,
        events_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Combines technical features, ratio metrics, macro indicators, and event features.
        Calculates future return target variables strictly shifted into future.
        """
        df = metals_df.copy().sort_values("date").reset_index(drop=True)

        if macro_df is not None and not macro_df.empty:
            df = pd.merge(df, macro_df, on="date", how="left")

        # Event Features (days_to_event, days_since_event, event_type)
        if events_df is not None and not events_df.empty:
            df = self._add_event_features(df, events_df)

        # Build Future Target Variables (Strictly Look-Ahead Protected)
        for h in self.forecast_horizons:
            # Shift close into future to create target
            df[f"future_close_{h}d"] = df["close"].shift(-h)
            df[f"future_return_{h}d"] = (df[f"future_close_{h}d"] - df["close"]) / df["close"]
            df[f"future_direction_{h}d"] = (df[f"future_return_{h}d"] > 0).astype(int)

        return df

    def _add_event_features(self, df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
        """Adds days_to_event and days_since_event strictly using calendar dates."""
        df["date_dt"] = pd.to_datetime(df["date"])
        events_df["event_date_dt"] = pd.to_datetime(events_df["date"])

        days_since = []
        days_to = []

        for dt in df["date_dt"]:
            past_events = events_df[events_df["event_date_dt"] <= dt]
            future_events = events_df[events_df["event_date_dt"] > dt]

            days_since.append(
                (dt - past_events["event_date_dt"].max()).days if not past_events.empty else 999
            )
            days_to.append(
                (future_events["event_date_dt"].min() - dt).days if not future_events.empty else 999
            )

        df["days_since_event"] = days_since
        df["days_to_event"] = days_to
        df = df.drop(columns=["date_dt"])
        return df
