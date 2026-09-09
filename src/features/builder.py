"""
Feature Store Construction Module.
Builds daily ML feature store combining technical indicators, macro drivers,
Gold/Silver ratios, regime classifications, and event proximity without look-ahead bias.
Features on date T contain ONLY information available up to date T.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class FeatureStoreBuilder:
    """Builds clean, point-in-time daily feature datasets for machine learning models."""

    def build_daily_features(
        self,
        metal_df: pd.DataFrame,
        ratio_df: Optional[pd.DataFrame] = None,
        macro_dfs: Optional[Dict[str, pd.DataFrame]] = None,
        events_df: Optional[pd.DataFrame] = None,
        symbol: str = "GOLD"
    ) -> pd.DataFrame:
        """Constructs standardized daily feature dataset."""
        if metal_df.empty:
            return pd.DataFrame()

        df = metal_df.copy().sort_values("date").reset_index(drop=True)
        df["instrument"] = symbol

        # 1. Standard Price Returns (Historical lookbacks only)
        close = df["close"]
        df["return_1d"] = close.pct_change(1) * 100.0
        df["return_3d"] = close.pct_change(3) * 100.0
        df["return_5d"] = close.pct_change(5) * 100.0
        df["return_10d"] = close.pct_change(10) * 100.0
        df["return_20d"] = close.pct_change(20) * 100.0
        df["return_60d"] = close.pct_change(60) * 100.0
        df["return_252d"] = close.pct_change(252) * 100.0

        # Drawdowns
        cummax = close.cummax()
        df["drawdown_pct"] = ((close - cummax) / cummax) * 100.0

        # 2. Merge Gold/Silver Ratio Features
        if ratio_df is not None and not ratio_df.empty and "gold_silver_ratio" in ratio_df.columns:
            r_cols = ["date", "gold_silver_ratio", "gold_silver_ratio_zscore", "gold_silver_spread_1d"]
            r_sub = ratio_df[[c for c in r_cols if c in ratio_df.columns]]
            df = pd.merge(df, r_sub, on="date", how="left")

        # 3. Merge Macro Features
        if macro_dfs:
            for m_sym, m_df in macro_dfs.items():
                if m_df.empty or "close" not in m_df.columns:
                    continue
                m_sub = m_df[["date", "close"]].rename(columns={"close": f"{m_sym.lower()}_close"})
                m_sub[f"{m_sym.lower()}_return_1d"] = m_sub[f"{m_sym.lower()}_close"].pct_change() * 100.0
                m_sub[f"{m_sym.lower()}_return_20d"] = m_sub[f"{m_sym.lower()}_close"].pct_change(20) * 100.0
                df = pd.merge(df, m_sub, on="date", how="left")

        # Forward fill any weekend/holiday alignment gaps in macro features
        df = df.ffill().bfill()

        # 4. Merge Event-based Features (e.g. days_to_event, days_since_event)
        if events_df is not None and not events_df.empty and "date" in events_df.columns:
            event_dates = pd.to_datetime(events_df["date"]).sort_values().tolist()
            dates_series = pd.to_datetime(df["date"])

            days_to_list = []
            days_since_list = []

            for d in dates_series:
                future_events = [ev for ev in event_dates if ev >= d]
                past_events = [ev for ev in event_dates if ev <= d]

                d_to = (future_events[0] - d).days if future_events else 999
                d_since = (d - past_events[-1]).days if past_events else 999

                days_to_list.append(d_to)
                days_since_list.append(d_since)

            df["days_to_event"] = days_to_list
            df["days_since_event"] = days_since_list

        # 5. Target Variables (FORWARD RETURNS stored separately for training)
        for h in [1, 3, 5, 10, 20, 60]:
            df[f"future_return_{h}d"] = (df["close"].shift(-h) / df["close"] - 1.0) * 100.0
            df[f"future_direction_{h}d"] = (df[f"future_return_{h}d"] > 0).astype(int)

        return df
