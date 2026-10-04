import pytest
import pandas as pd
import numpy as np
from src.scoring.composite import MarketScorer
from src.models.baseline import ForecastingModelPipeline
from src.backtesting.engine import BacktestEngine
from src.evaluation.performance import ForecastEvaluator

def test_market_scorer():
    row = pd.Series({"rsi_14": 65.0, "dist_sma_200": 0.05, "macd": 1.2, "return_20d": 0.04, "volatility_20d": 0.15, "gold_silver_ratio_zscore": 1.1})
    macro = {"explainable_components": {"dxy_return_20d": -1.5, "us10y_change_20d": -0.1}}
    analogue = {"forward_statistics": {"5d": {"prob_positive": 0.65}}}
    weights = {"technical": 0.25, "macro": 0.20}

    res = MarketScorer.calculate_scores(row, macro, analogue, weights)
    assert "composite_score" in res
    assert 0 <= res["composite_score"] <= 100
    assert "sub_scores" in res
    assert res["sub_scores"]["technical_score"] > 50

def test_forecasting_pipeline_walk_forward():
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "f1": np.random.normal(0, 1, 100),
        "f2": np.random.normal(0, 1, 100),
        "future_return_5d": np.random.normal(0.001, 0.02, 100)
    })
    pipe = ForecastingModelPipeline(feature_cols=["f1", "f2"])
    res = pipe.train_and_evaluate_walk_forward(df, target_col="future_return_5d", min_train_size=40, step_size=10)
    assert res["status"] == "success"
    assert "directional_accuracy" in res
    assert res["samples_evaluated"] > 0

def test_backtest_engine():
    dates = pd.date_range("2024-01-01", periods=50, freq="B")
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "price": np.linspace(100, 120, 50),
        "composite_score": [30]*10 + [65]*20 + [35]*20
    })
    engine = BacktestEngine(initial_capital=100000.0)
    res = engine.run_signal_backtest(df, signal_col="composite_score", buy_threshold=60, sell_threshold=40)
    assert res["trade_count"] > 0
    assert "final_portfolio_value" in res

def test_forecast_evaluator():
    df = pd.DataFrame({
        "predicted_return": [0.01, -0.02, 0.015, -0.01],
        "actual_return": [0.012, -0.01, -0.005, -0.02],
        "market_regime": ["Bullish", "Bearish", "Bullish", "Bearish"]
    })
    res = ForecastEvaluator.evaluate_predictions(df)
    assert res["total_evaluated"] == 4
    assert res["directional_accuracy"] == 0.75
