"""
Indian Market Unit Conversion and Precious Metals Processing Module.
Supports USD/oz to INR/10g (Gold) and INR/kg (Silver) conversion,
recording exact exchange rate, conversion timestamp, and duty assumptions.
"""

from datetime import datetime, timezone
from typing import Dict, Any, Optional
import pandas as pd


# Constants
OZ_TO_GRAMS = 31.1034768
GRAMS_PER_10G = 10.0
GRAMS_PER_KG = 1000.0


class IndianMarketConverter:
    """Performs transparent conversions between USD spot gold/silver and Indian spot/futures measurements."""

    def __init__(self, custom_duty_pct: float = 0.06):
        self.custom_duty_pct = custom_duty_pct

    def convert_gold_usd_to_inr_10g(self, gold_usd_per_oz: float, usdinr_rate: float) -> float:
        """Converts USD/oz Gold price to INR per 10 grams (including duty factor)."""
        price_per_gram_usd = gold_usd_per_oz / OZ_TO_GRAMS
        price_per_10g_usd = price_per_gram_usd * GRAMS_PER_10G
        price_per_10g_inr = price_per_10g_usd * usdinr_rate * (1.0 + self.custom_duty_pct)
        return round(price_per_10g_inr, 2)

    def convert_silver_usd_to_inr_kg(self, silver_usd_per_oz: float, usdinr_rate: float) -> float:
        """Converts USD/oz Silver price to INR per kg (including duty factor)."""
        price_per_gram_usd = silver_usd_per_oz / OZ_TO_GRAMS
        price_per_kg_usd = price_per_gram_usd * GRAMS_PER_KG
        price_per_kg_inr = price_per_kg_usd * usdinr_rate * (1.0 + self.custom_duty_pct)
        return round(price_per_kg_inr, 2)

    def process_indian_metals_dataset(
        self,
        gold_df: pd.DataFrame,
        silver_df: pd.DataFrame,
        usdinr_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merges USD Gold/Silver with USD/INR rate and derives official INR Indian market series."""
        merged = pd.merge(
            gold_df[["date", "close"]].rename(columns={"close": "gold_usd"}),
            silver_df[["date", "close"]].rename(columns={"close": "silver_usd"}),
            on="date",
            how="inner",
        )
        merged = pd.merge(
            merged,
            usdinr_df[["date", "close"]].rename(columns={"close": "usdinr"}),
            on="date",
            how="inner",
        )

        merged["gold_inr_10g"] = merged.apply(
            lambda r: self.convert_gold_usd_to_inr_10g(r["gold_usd"], r["usdinr"]), axis=1
        )
        merged["silver_inr_kg"] = merged.apply(
            lambda r: self.convert_silver_usd_to_inr_kg(r["silver_usd"], r["usdinr"]), axis=1
        )
        merged["conversion_duty_pct"] = self.custom_duty_pct
        merged["conversion_timestamp"] = datetime.now(timezone.utc).isoformat()

        return merged
