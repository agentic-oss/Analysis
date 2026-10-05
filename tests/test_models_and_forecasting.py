"""Test suite for analogue search engine, features, scoring, forecasting, and backtesting."""

import numpy as np
import pandas as pd
from src.patterns.analogue_engine import AnalogueEngine
from src.features.feature_builder import FeatureBuilder
from src.scoring.composite_scorer import CompositeScorer
from src.models.forecasting_models import MetalsForecastingModels
from src.forecasting.probabilistic_engine import ProbabilisticForecaster
from src.backtesting.backtest_engine import BacktestEngine


def test_analogue_engine_and_features():
    dates = pd.date_range('2025-01-01', '2025-10-01', freq='B')
    arr = np.arange(len(dates))
    df = pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d'),
        'symbol': 'GOLD',
        'close': 2000.0 * (1.0002 ** arr),
        'rsi_14': 50.0 + (arr % 20),
        'macd': 1.0,
        'volatility_20d': 0.12,
        'distance_sma_50': 0.01,
        'distance_sma_200': 0.03,
        'return_20d': 0.01
    })

    feat_df = FeatureBuilder.build_daily_features(df, {})
    assert 'future_return_10d' in feat_df.columns

    engine = AnalogueEngine(top_k=5)
    target_d = feat_df.iloc[-5]['date']
    analogue_res = engine.find_analogues(feat_df, target_d, 'GOLD')
    assert analogue_res['sample_count'] <= 5


def test_composite_scorer():
    scorer = CompositeScorer()
    res = scorer.compute_scores(
        tech_indicators={'rsi_14': 55.0, 'distance_sma_50': 0.01, 'distance_sma_200': 0.05, 'macd_hist': 1.0, 'roc_10': 2.0},
        macro_correlations={'DXY': {'corr_60d': -0.4, 'return_1d': -0.001}},
        ratio_metrics={'z_score_252d': 0.2},
        analogue_stats={'forward_stats': {'10d': {'positive_probability': 0.60}}}
    )
    assert 0 <= res['composite_score'] <= 100


def test_backtest_engine():
    dates = pd.date_range('2025-01-01', '2025-06-01', freq='B')
    arr = np.arange(len(dates))
    df = pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d'),
        'close': 100.0 + arr * 0.2,
        'signal': 1
    })
    bt = BacktestEngine(initial_capital=100000.0)
    res = bt.run_signal_backtest(df, 'signal', 'GOLD')
    assert res['total_return_pct'] > 0
