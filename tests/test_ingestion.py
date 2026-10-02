import pytest
import pandas as pd
from src.ingestion.converter import UnitCurrencyConverter

def test_unit_currency_converter():
    converter = UnitCurrencyConverter()

    # USD/oz to INR/10g
    # Gold $2500/oz at 83.5 USDINR -> (2500 * 83.5) / 3.11034768 = 67114.68 INR/10g
    inr_10g = converter.convert_gold_usd_oz_to_inr_10g(2500.0, 83.5)
    assert abs(inr_10g - 67114.68) < 1.0

    # Silver $30/oz at 83.5 USDINR -> 30 * 83.5 * 32.1507466 = 80537.62 INR/kg
    inr_kg = converter.convert_silver_usd_oz_to_inr_kg(30.0, 83.5)
    assert abs(inr_kg - 80537.62) < 1.0

    # Reverse conversions
    usd_gold = converter.convert_gold_inr_10g_to_usd_oz(inr_10g, 83.5)
    assert abs(usd_gold - 2500.0) < 0.01

    usd_silver = converter.convert_silver_inr_kg_to_usd_oz(inr_kg, 83.5)
    assert abs(usd_silver - 30.0) < 0.01
