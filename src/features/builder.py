import logging
import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.indicators.technical import TechnicalIndicators

logger = logging.getLogger(__name__)


class FeatureStoreBuilder:
    """Builds ML-ready daily feature store datasets with strict target separation and macro event features."""

    def __init__(self, horizons: List[int] = [1, 3, 5, 10, 20, 60]):
        self.horizons = horizons

    def create_features_for_instrument(
        self,
        instrument_df: pd.DataFrame,
        macro_datasets: Dict[str, pd.DataFrame],
        events_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Calculates features available at time T and forward targets for training/backtesting."""
        if instrument_df.empty:
            return pd.DataFrame()

        df = TechnicalIndicators.calculate_all(instrument_df)
        df = df.sort_values("date").reset_index(drop=True)

        # Add macro features available at time T
        for sym, mdf in macro_datasets.items():
            if mdf.empty or "close" not in mdf.columns:
                continue
            m = mdf[["date", "close"]].copy().sort_values("date").reset_index(drop=True)
            col_name = f"macro_{sym.lower()}_close"
            m = m.rename(columns={"close": col_name})
            m[f"macro_{sym.lower()}_ret_1d"] = m[col_name].pct_change()

            df = pd.merge(df, m, on="date", how="left")

        # Macro Event Features (days_to_event, days_since_event)
        if events_df is not None and not events_df.empty and "date" in events_df.columns:
            df = self._add_event_features(df, events_df)
        else:
            df["days_to_next_event"] = 999
            df["days_since_prev_event"] = 999
            df["event_importance"] = "none"

        # Separate Forward Return Targets (strictly looking ahead into the future)
        for h in self.horizons:
            # target return: (close[t+h] - close[t]) / close[t]
            df[f"future_return_{h}d"] = (df["close"].shift(-h) - df["close"]) / df["close"]
            df[f"future_direction_{h}d"] = (df[f"future_return_{h}d"] > 0).astype(int)

        return df

    def _add_event_features(self, df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
        df["date"] = pd.to_datetime(df["date"])
        events_df["date"] = pd.to_datetime(events_df["date"])

        event_dates = sorted(events_df["date"].unique())
        days_to_next = []
        days_since_prev = []

        for current_date in df["date"]:
            next_events = [e for e in event_dates if e > current_date]
            prev_events = [e for e in event_dates if e <= current_date]

            days_to_next.append((next_events[0] - current_date).days if next_events else 999)
            days_since_prev.append((current_date - prev_events[-1]).days if prev_events else 999)

        df["days_to_next_event"] = days_to_next
        df["days_since_prev_event"] = days_since_prev
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        return df
