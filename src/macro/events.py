import pandas as pd
import numpy as np

class MacroEventTracker:
    """Tracks major economic calendar events (FOMC, CPI, PCE, NFP, RBI) and calculates event proximity features."""

    def __init__(self, events_list: list[dict] = None):
        # Default major recurrent macro event schedule or custom events
        self.events_list = events_list or []

    def calculate_event_features(self, date_str: str) -> dict:
        """Calculates days_to_next_event and days_since_last_event for date_str."""
        if not self.events_list:
            return {
                "days_to_event": np.nan,
                "days_since_event": np.nan,
                "event_type": "None",
                "event_importance": "Low"
            }

        curr_dt = pd.to_datetime(date_str)
        events_df = pd.DataFrame(self.events_list)
        events_df["event_date"] = pd.to_datetime(events_df["date"])

        future_events = events_df[events_df["event_date"] >= curr_dt].sort_values("event_date")
        past_events = events_df[events_df["event_date"] < curr_dt].sort_values("event_date", ascending=False)

        days_to = (future_events.iloc[0]["event_date"] - curr_dt).days if not future_events.empty else np.nan
        days_since = (curr_dt - past_events.iloc[0]["event_date"]).days if not past_events.empty else np.nan
        next_type = future_events.iloc[0]["event_type"] if not future_events.empty else "None"
        next_importance = future_events.iloc[0].get("importance", "Medium") if not future_events.empty else "Low"

        return {
            "days_to_event": days_to,
            "days_since_event": days_since,
            "event_type": next_type,
            "event_importance": next_importance
        }
