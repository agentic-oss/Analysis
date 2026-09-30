import logging
from datetime import datetime
from typing import Optional, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)

# Standard conversion constants
OUNCES_PER_GRAM = 0.0321507466
GRAMS_PER_OUNCE = 31.1034768


def convert_usd_oz_to_inr_10g(price_usd_oz: float, usdinr_rate: float) -> Optional[float]:
    """
    Converts price from USD/oz to INR/10g.
    Formula: (price_usd_oz / 31.1034768) * 10 * usdinr_rate
    """
    if price_usd_oz is None or usdinr_rate is None or usdinr_rate <= 0:
        return None
    price_per_gram_usd = price_usd_oz / GRAMS_PER_OUNCE
    price_10g_usd = price_per_gram_usd * 10.0
    return price_10g_usd * usdinr_rate


def convert_usd_oz_to_inr_kg(price_usd_oz: float, usdinr_rate: float) -> Optional[float]:
    """
    Converts price from USD/oz to INR/kg.
    Formula: (price_usd_oz / 31.1034768) * 1000 * usdinr_rate
    """
    if price_usd_oz is None or usdinr_rate is None or usdinr_rate <= 0:
        return None
    price_per_gram_usd = price_usd_oz / GRAMS_PER_OUNCE
    price_kg_usd = price_per_gram_usd * 1000.0
    return price_kg_usd * usdinr_rate


def process_indian_currency_conversions(
    metals_df: pd.DataFrame, usdinr_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges metals data with USDINR exchange rates and calculates explicit INR converted prices.
    Retains original prices, exchange rates, and conversion timestamps.
    """
    if metals_df.empty or usdinr_df.empty:
        return metals_df

    # Prepare exchange rate reference
    rates = usdinr_df[["date", "close"]].rename(columns={"close": "usdinr_rate"})

    df = pd.merge(metals_df, rates, on="date", how="left")
    df["usdinr_rate"] = df["usdinr_rate"].ffill().bfill()

    now_iso = datetime.utcnow().isoformat()
    df["conversion_timestamp"] = now_iso

    # Convert prices based on instrument units
    if "unit" in df.columns:
        gold_mask = (df["unit"] == "oz") & (df["symbol"].str.contains("GOLD", na=False))
        silver_mask = (df["unit"] == "oz") & (df["symbol"].str.contains("SILVER", na=False))

        df.loc[gold_mask, "price_inr_10g"] = df.loc[gold_mask].apply(
            lambda row: convert_usd_oz_to_inr_10g(row["close"], row["usdinr_rate"]), axis=1
        )
        df.loc[silver_mask, "price_inr_kg"] = df.loc[silver_mask].apply(
            lambda row: convert_usd_oz_to_inr_kg(row["close"], row["usdinr_rate"]), axis=1
        )

    return df
