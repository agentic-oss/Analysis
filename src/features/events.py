import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List


class EventFeatureBuilder:
    """Builds macro economic event proximity features."""

    @staticmethod
    def attach_event_features(
        price_df: pd.DataFrame, events_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Attaches days_to_event, days_since_event, event_type, event_importance
        for each date in price_df.
        """
        if price_df is None or price_df.empty:
            return price_df

        df = price_df.copy()
        if events_df is None or events_df.empty or "date" not in events_df.columns:
            df["days_to_event"] = 999
            df["days_since_event"] = 999
            df["event_type"] = "none"
            df["event_importance"] = "low"
            return df

        df["Date"] = pd.to_datetime(df["Date"])
        events = events_df.copy()
        events["date"] = pd.to_datetime(events["date"])

        days_to_list = []
        days_since_list = []
        event_types = []
        event_importances = []

        for _, row in df.iterrows():
            d = row["Date"]

            # Future events
            future_events = events[events["date"] >= d]
            if not future_events.empty:
                next_event = future_events.iloc[0]
                days_to = (next_event["date"] - d).days
                e_type = next_event.get("type", "macro")
                e_imp = next_event.get("importance", "medium")
            else:
                days_to = 999
                e_type = "none"
                e_imp = "low"

            # Past events
            past_events = events[events["date"] <= d]
            if not past_events.empty:
                last_event = past_events.iloc[-1]
                days_since = (d - last_event["date"]).days
            else:
                days_since = 999

            days_to_list.append(days_to)
            days_since_list.append(days_since)
            event_types.append(e_type)
            event_importances.append(e_imp)

        df["days_to_event"] = days_to_list
        df["days_since_event"] = days_since_list
        df["event_type"] = event_types
        df["event_importance"] = event_importances

        return df
