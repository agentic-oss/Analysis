"""Test suite for converters and data ingestion."""

import pytest
from src.ingestion.converters import (
    convert_gold_price,
    convert_silver_price,
    convert_inr_gold_to_usd_oz,
    convert_inr_silver_to_usd_oz
)


def test_convert_gold_price():
    # 2000 USD/oz at 80 USDINR => 160000 INR/oz
    # 160000 / 31.1034768 * 10 = ~5144.12 per gram * 10 = 51441.2
    res = convert_gold_price(2000.0, 80.0, "10g")
    assert pytest.approx(res, 0.1) == 51441.2

    res_kg = convert_gold_price(2000.0, 80.0, "kg")
    assert pytest.approx(res_kg, 1.0) == 514412.0


def test_convert_silver_price():
    # 25 USD/oz at 80 USDINR => 2000 INR/oz
    # 2000 / 31.1034768 * 1000 = ~64301.8 INR/kg
    res = convert_silver_price(25.0, 80.0, "kg")
    assert pytest.approx(res, 1.0) == 64301.8


def test_roundtrip_conversions():
    price_usd = 2100.0
    rate = 83.5
    price_inr_10g = convert_gold_price(price_usd, rate, "10g")
    reconstructed_usd = convert_inr_gold_to_usd_oz(price_inr_10g, rate)
    assert pytest.approx(reconstructed_usd, 0.01) == price_usd
