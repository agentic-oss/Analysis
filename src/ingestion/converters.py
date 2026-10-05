"""Unit and Currency Conversion Utilities.

Converts precious metal prices between USD/oz, INR/10g, INR/kg and handles currency conversions.
"""

# Constants
TROY_OZ_TO_GRAMS = 31.1034768


def convert_gold_price(
    price_usd_per_oz: float,
    usdinr_rate: float,
    target_unit: str = "10g"
) -> float:
    """
    Convert Gold price from USD/oz to INR per target unit (e.g. '10g' or 'oz').
    """
    if price_usd_per_oz is None or usdinr_rate is None or usdinr_rate <= 0:
        return 0.0

    price_inr_per_oz = price_usd_per_oz * usdinr_rate

    if target_unit == "10g":
        # Price per gram = price_inr_per_oz / TROY_OZ_TO_GRAMS
        # Price per 10g = (price_inr_per_oz / TROY_OZ_TO_GRAMS) * 10
        return (price_inr_per_oz / TROY_OZ_TO_GRAMS) * 10.0
    elif target_unit == "g":
        return price_inr_per_oz / TROY_OZ_TO_GRAMS
    elif target_unit == "kg":
        return (price_inr_per_oz / TROY_OZ_TO_GRAMS) * 1000.0
    elif target_unit == "oz":
        return price_inr_per_oz
    else:
        raise ValueError(f"Unsupported target_unit: {target_unit}")


def convert_silver_price(
    price_usd_per_oz: float,
    usdinr_rate: float,
    target_unit: str = "kg"
) -> float:
    """
    Convert Silver price from USD/oz to INR per target unit (e.g. 'kg' or '10g').
    """
    if price_usd_per_oz is None or usdinr_rate is None or usdinr_rate <= 0:
        return 0.0

    price_inr_per_oz = price_usd_per_oz * usdinr_rate

    if target_unit == "kg":
        # Price per 1kg = (price_inr_per_oz / TROY_OZ_TO_GRAMS) * 1000
        return (price_inr_per_oz / TROY_OZ_TO_GRAMS) * 1000.0
    elif target_unit == "10g":
        return (price_inr_per_oz / TROY_OZ_TO_GRAMS) * 10.0
    elif target_unit == "oz":
        return price_inr_per_oz
    else:
        raise ValueError(f"Unsupported target_unit: {target_unit}")


def convert_inr_gold_to_usd_oz(
    price_inr_per_10g: float,
    usdinr_rate: float
) -> float:
    """Convert Gold price from INR/10g to USD/oz."""
    if price_inr_per_10g is None or usdinr_rate is None or usdinr_rate <= 0:
        return 0.0
    price_inr_per_gram = price_inr_per_10g / 10.0
    price_inr_per_oz = price_inr_per_gram * TROY_OZ_TO_GRAMS
    return price_inr_per_oz / usdinr_rate


def convert_inr_silver_to_usd_oz(
    price_inr_per_kg: float,
    usdinr_rate: float
) -> float:
    """Convert Silver price from INR/kg to USD/oz."""
    if price_inr_per_kg is None or usdinr_rate is None or usdinr_rate <= 0:
        return 0.0
    price_inr_per_gram = price_inr_per_kg / 1000.0
    price_inr_per_oz = price_inr_per_gram * TROY_OZ_TO_GRAMS
    return price_inr_per_oz / usdinr_rate
