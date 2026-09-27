import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Constants for conversion
TROY_OUNCE_TO_GRAMS = 31.1034768
GRAMS_PER_10G = 10.0
GRAMS_PER_KG = 1000.0

def convert_usd_oz_to_inr_10g(price_usd_oz: float, usdinr: float) -> float:
    """Converts USD per troy ounce to INR per 10 grams."""
    if pd.isna(price_usd_oz) or pd.isna(usdinr) or price_usd_oz <= 0 or usdinr <= 0:
        return np.nan
    price_usd_per_gram = price_usd_oz / TROY_OUNCE_TO_GRAMS
    price_inr_per_gram = price_usd_per_gram * usdinr
    return price_inr_per_gram * GRAMS_PER_10G

def convert_usd_oz_to_inr_kg(price_usd_oz: float, usdinr: float) -> float:
    """Converts USD per troy ounce to INR per kilogram."""
    if pd.isna(price_usd_oz) or pd.isna(usdinr) or price_usd_oz <= 0 or usdinr <= 0:
        return np.nan
    price_usd_per_gram = price_usd_oz / TROY_OUNCE_TO_GRAMS
    price_inr_per_gram = price_usd_per_gram * usdinr
    return price_inr_per_gram * GRAMS_PER_KG

def process_and_align_datasets(data_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combines individual instrument DataFrames on 'date', preserving OHLCV columns for each instrument,
    performs INR price conversions, and returns a clean, aligned master daily dataframe.
    """
    if not data_map:
        return pd.DataFrame()

    merged = None
    for symbol, df in data_map.items():
        if df.empty:
            continue

        # Keep OHLCV columns if present
        cols_to_keep = ["date"]
        rename_dict = {}
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                cols_to_keep.append(col)
                rename_dict[col] = f"{symbol}_{col}"

        sub = df[cols_to_keep].rename(columns=rename_dict)
        if merged is None:
            merged = sub
        else:
            merged = pd.merge(merged, sub, on="date", how="outer")

    if merged is None or merged.empty:
        return pd.DataFrame()

    merged = merged.sort_values("date").reset_index(drop=True)
    merged = merged.ffill().bfill()

    # Calculate converted INR prices where USD prices exist
    if "USDINR_close" in merged.columns:
        if "GOLD_close" in merged.columns:
            merged["GOLD_INR_calc_10g"] = merged.apply(
                lambda r: convert_usd_oz_to_inr_10g(r["GOLD_close"], r["USDINR_close"]), axis=1
            )
        if "SILVER_close" in merged.columns:
            merged["SILVER_INR_calc_kg"] = merged.apply(
                lambda r: convert_usd_oz_to_inr_kg(r["SILVER_close"], r["USDINR_close"]), axis=1
            )

    return merged
