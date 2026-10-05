"""Test suite for indicators, ratio analysis, and regime detection."""

import numpy as np
import pandas as pd
from src.indicators.technical import TechnicalIndicators
from src.ratios.ratio_analyzer import GoldSilverRatioAnalyzer
from src.regimes.regime_detector import RegimeDetector


def test_technical_indicators_calculation():
    dates = pd.date_range('2025-01-01', '2025-06-01', freq='B')
    arr = np.arange(len(dates))
    df = pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d'),
        'close': 100.0 + arr * 0.5,
        'open': 100.0 + arr * 0.5,
        'high': 101.0 + arr * 0.5,
        'low': 99.0 + arr * 0.5,
        'volume': 1000
    })

    res = TechnicalIndicators.calculate_indicators(df)
    assert 'sma_20' in res.columns
    assert 'rsi_14' in res.columns
    assert 'macd' in res.columns
    assert 'atr_14' in res.columns
    assert 'bb_upper' in res.columns
    assert len(res) == len(dates)


def test_gold_silver_ratio_analyzer():
    dates = pd.date_range('2025-01-01', '2025-03-01', freq='B')
    gold_df = pd.DataFrame({'date': dates.strftime('%Y-%m-%d'), 'close': 2000.0})
    silver_df = pd.DataFrame({'date': dates.strftime('%Y-%m-%d'), 'close': 25.0})

    ratio_res = GoldSilverRatioAnalyzer.calculate_ratio_metrics(gold_df, silver_df)
    assert ratio_res['current_ratio'] == 80.0
    assert ratio_res['percentile_rank'] >= 0.0


def test_regime_detector():
    detector = RegimeDetector()
    macro_reg = detector.detect_macro_regime(
        rates_change_20d=0.05,
        dxy_change_20d=0.03,
        vix_level=15.0,
        oil_change_20d=0.06,
        sp500_change_20d=-0.02
    )
    assert macro_reg['primary_regime'] in ["Inflationary", "Tightening", "Risk-Off", "Risk-On", "Easing", "Neutral"]
