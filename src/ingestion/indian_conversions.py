import datetime
import logging
import pandas as pd
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Constants for unit conversions
TROY_OUNCE_TO_GRAMS = 31.1034768
GRAMS_PER_10G = 10.0
GRAMS_PER_KG = 1000.0

def convert_usd_oz_to_inr_10g(usd_oz: float, usdinr_rate: float) -> Optional[float]:
    """Convert USD per troy ounce to INR per 10 grams."""
    if pd.isna(usd_oz) or pd.isna(usdinr_rate) or usdinr_rate <= 0:
        return None
    # USD/oz -> USD/gram -> INR/gram -> INR/10g
    usd_per_gram = usd_oz / TROY_OUNCE_TO_GRAMS
    inr_per_gram = usd_per_gram * usdinr_rate
    return inr_per_gram * GRAMS_PER_10G

def convert_usd_oz_to_inr_kg(usd_oz: float, usdinr_rate: float) -> Optional[float]:
    """Convert USD per troy ounce to INR per kilogram."""
    if pd.isna(usd_oz) or pd.isna(usdinr_rate) or usdinr_rate <= 0:
        return None
    # USD/oz -> USD/gram -> INR/gram -> INR/kg
    usd_per_gram = usd_oz / TROY_OUNCE_TO_GRAMS
    inr_per_gram = usd_per_gram * usdinr_rate
    return inr_per_gram * GRAMS_PER_KG

def enrich_indian_conversions(
    metals_df: pd.DataFrame,
    usdinr_df: pd.DataFrame,
    symbol: str = "GOLD"
) -> pd.DataFrame:
    """
    Enriches metals price dataframe with calculated INR prices using exact USDINR exchange rates.
    Stores original price, converted INR price, exchange rate used, and conversion timestamp.
    """
    df = metals_df.copy()
    if usdinr_df.empty:
        df['usdinr_rate'] = None
        df['converted_inr'] = None
        df['conversion_timestamp'] = None
        return df

    # Join on index (Date)
    usdinr_rates = usdinr_df['Close'].rename('usdinr_rate')
    df = df.join(usdinr_rates, how='left')
    df['usdinr_rate'] = df['usdinr_rate'].ffill().bfill()

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if symbol in ["GOLD", "GOLD_FUTURES"]:
        df['converted_inr'] = df.apply(
            lambda r: convert_usd_oz_to_inr_10g(r['Close'], r['usdinr_rate']), axis=1
        )
        df['conversion_unit'] = "INR/10g"
    elif symbol in ["SILVER", "SILVER_FUTURES"]:
        df['converted_inr'] = df.apply(
            lambda r: convert_usd_oz_to_inr_kg(r['Close'], r['usdinr_rate']), axis=1
        )
        df['conversion_unit'] = "INR/kg"
    else:
        df['converted_inr'] = df['Close']
        df['conversion_unit'] = "native"

    df['conversion_timestamp'] = now_iso
    return df
