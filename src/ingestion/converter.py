from datetime import datetime, timezone
import pandas as pd
import numpy as np

# Conversion Constants
TROY_OZ_TO_GRAMS = 31.1034768
TROY_OZ_TO_10G = 3.11034768
KG_TO_TROY_OZ = 32.1507466

class UnitCurrencyConverter:
    """Handles unit and currency conversions for precious metals and macro assets."""

    @staticmethod
    def convert_gold_usd_oz_to_inr_10g(usd_oz_price: float, usdinr_rate: float) -> float:
        if pd.isna(usd_oz_price) or pd.isna(usdinr_rate) or usdinr_rate <= 0:
            return np.nan
        return (usd_oz_price * usdinr_rate) / TROY_OZ_TO_10G

    @staticmethod
    def convert_silver_usd_oz_to_inr_kg(usd_oz_price: float, usdinr_rate: float) -> float:
        if pd.isna(usd_oz_price) or pd.isna(usdinr_rate) or usdinr_rate <= 0:
            return np.nan
        return usd_oz_price * usdinr_rate * KG_TO_TROY_OZ

    @staticmethod
    def convert_gold_inr_10g_to_usd_oz(inr_10g_price: float, usdinr_rate: float) -> float:
        if pd.isna(inr_10g_price) or pd.isna(usdinr_rate) or usdinr_rate <= 0:
            return np.nan
        return (inr_10g_price * TROY_OZ_TO_10G) / usdinr_rate

    @staticmethod
    def convert_silver_inr_kg_to_usd_oz(inr_kg_price: float, usdinr_rate: float) -> float:
        if pd.isna(inr_kg_price) or pd.isna(usdinr_rate) or usdinr_rate <= 0:
            return np.nan
        return inr_kg_price / (usdinr_rate * KG_TO_TROY_OZ)

    def process_metal_dataframe(self, df: pd.DataFrame, usdinr_series: pd.Series) -> pd.DataFrame:
        """
        Takes a dataframe with precious metals prices, merges USD/INR, and adds converted prices.
        Maintains both original market price and converted INR/USD prices, recording exact conversion details.
        """
        if df.empty:
            return df

        df = df.copy()
        now_iso = datetime.now(timezone.utc).isoformat()

        # Ensure date alignment
        if "date" in df.columns and usdinr_series is not None and not usdinr_series.empty:
            df["usdinr_rate"] = df["date"].map(usdinr_series)
        else:
            df["usdinr_rate"] = np.nan

        df["conversion_timestamp"] = now_iso

        converted_prices = []
        target_units = []
        target_currencies = []

        for idx, row in df.iterrows():
            sym = row.get("symbol", "")
            curr = row.get("currency", "USD")
            unit = row.get("unit", "oz")
            close_price = row.get("close", np.nan)
            rate = row.get("usdinr_rate", np.nan)

            if "GOLD" in str(sym) and curr == "USD" and unit == "oz":
                conv = self.convert_gold_usd_oz_to_inr_10g(close_price, rate)
                converted_prices.append(conv)
                target_currencies.append("INR")
                target_units.append("10g")
            elif "SILVER" in str(sym) and curr == "USD" and unit == "oz":
                conv = self.convert_silver_usd_oz_to_inr_kg(close_price, rate)
                converted_prices.append(conv)
                target_currencies.append("INR")
                target_units.append("kg")
            elif "GOLD" in str(sym) and curr == "INR" and unit == "10g":
                conv = self.convert_gold_inr_10g_to_usd_oz(close_price, rate)
                converted_prices.append(conv)
                target_currencies.append("USD")
                target_units.append("oz")
            elif "SILVER" in str(sym) and curr == "INR" and unit == "kg":
                conv = self.convert_silver_inr_kg_to_usd_oz(close_price, rate)
                converted_prices.append(conv)
                target_currencies.append("USD")
                target_units.append("oz")
            else:
                converted_prices.append(np.nan)
                target_currencies.append(curr)
                target_units.append(unit)

        df["converted_price"] = converted_prices
        df["target_currency"] = target_currencies
        df["target_unit"] = target_units

        return df
