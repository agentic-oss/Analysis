"""Macro economic events and data ingestion module."""

from typing import List, Dict, Any
from datetime import datetime


class MacroEventProvider:
    """Provides economic calendar and macro central bank event schedules."""

    def __init__(self):
        # Deterministic / Scheduled macro calendar
        self.events_schedule = [
            {"date": "2026-01-14", "event": "US_CPI", "category": "inflation", "importance": "high", "country": "US"},
            {"date": "2026-01-28", "event": "FOMC_DECISION", "category": "central_bank", "importance": "high", "country": "US"},
            {"date": "2026-02-06", "event": "RBI_POLICY", "category": "central_bank", "importance": "high", "country": "IN"},
            {"date": "2026-02-13", "event": "US_CPI", "category": "inflation", "importance": "high", "country": "US"},
            {"date": "2026-02-27", "event": "US_PCE", "category": "inflation", "importance": "high", "country": "US"},
            {"date": "2026-03-06", "event": "US_NFP", "category": "employment", "importance": "high", "country": "US"},
            {"date": "2026-03-18", "event": "FOMC_DECISION", "category": "central_bank", "importance": "high", "country": "US"},
            {"date": "2026-04-08", "event": "RBI_POLICY", "category": "central_bank", "importance": "high", "country": "IN"},
        ]

    def get_events_for_date(self, date_str: str) -> List[Dict[str, Any]]:
        """Return scheduled macro events occurring on date_str."""
        return [e for e in self.events_schedule if e["date"] == date_str]

    def get_days_to_next_event(self, current_date_str: str, event_type: str) -> int:
        """Calculate days remaining until the next occurrence of event_type."""
        curr_dt = datetime.strptime(current_date_str, "%Y-%m-%d")
        future_dates = [
            datetime.strptime(e["date"], "%Y-%m-%d")
            for e in self.events_schedule
            if e["event"] == event_type and datetime.strptime(e["date"], "%Y-%m-%d") >= curr_dt
        ]
        if not future_dates:
            return 999
        min_future = min(future_dates)
        return (min_future - curr_dt).days

    def get_days_since_last_event(self, current_date_str: str, event_type: str) -> int:
        """Calculate days passed since the last occurrence of event_type."""
        curr_dt = datetime.strptime(current_date_str, "%Y-%m-%d")
        past_dates = [
            datetime.strptime(e["date"], "%Y-%m-%d")
            for e in self.events_schedule
            if e["event"] == event_type and datetime.strptime(e["date"], "%Y-%m-%d") <= curr_dt
        ]
        if not past_dates:
            return 999
        max_past = max(past_dates)
        return (curr_dt - max_past).days
