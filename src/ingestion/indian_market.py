from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple
import pandas as pd

TROY_OZ_TO_GRAMS = 31.1034768

def convert_usd_oz_to_inr_10g(usd_per_oz: float, usdinr_rate: float) -> float:
    """Convert USD/oz price to INR per 10g."""
    usd_per_gram = usd_per_oz / TROY_OZ_TO_GRAMS
    inr_per_gram = usd_per_gram * usdinr_rate
    return round(inr_per_gram * 10.0, 2)


def convert_usd_oz_to_inr_kg(usd_per_oz: float, usdinr_rate: float) -> float:
    """Convert USD/oz price to INR per kg."""
    usd_per_gram = usd_per_oz / TROY_OZ_TO_GRAMS
    inr_per_gram = usd_per_gram * usdinr_rate
    return round(inr_per_gram * 1000.0, 2)


def process_indian_price_dataframe(
    gold_df: pd.DataFrame,
    silver_df: pd.DataFrame,
    usdinr_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Given gold (USD/oz), silver (USD/oz), and USD/INR exchange rate DataFrames,
    calculate and attach INR converted prices, exchange rate used, and conversion timestamp.
    """
    gold_res = gold_df.copy()
    silver_res = silver_df.copy()

    # Align by Date
    if "Date" in gold_res.columns and "Date" in usdinr_df.columns:
        usdinr_map = dict(zip(pd.to_datetime(usdinr_df["Date"]).dt.date, usdinr_df["Close"]))

        # Gold INR
        gold_dates = pd.to_datetime(gold_res["Date"]).dt.date
        gold_res["USDINR_rate"] = gold_dates.map(usdinr_map).ffill().bfill().fillna(83.0)
        gold_res["Close_INR_10g"] = [
            convert_usd_oz_to_inr_10g(p, r)
            for p, r in zip(gold_res["Close"], gold_res["USDINR_rate"])
        ]
        gold_res["conversion_timestamp"] = datetime.now(timezone.utc).isoformat()

        # Silver INR
        silver_dates = pd.to_datetime(silver_res["Date"]).dt.date
        silver_res["USDINR_rate"] = silver_dates.map(usdinr_map).ffill().bfill().fillna(83.0)
        silver_res["Close_INR_kg"] = [
            convert_usd_oz_to_inr_kg(p, r)
            for p, r in zip(silver_res["Close"], silver_res["USDINR_rate"])
        ]
        silver_res["conversion_timestamp"] = datetime.now(timezone.utc).isoformat()

    return gold_res, silver_res
