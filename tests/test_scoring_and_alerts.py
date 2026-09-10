import pytest
import pandas as pd
from src.scoring.scoring import TransparentScoringSystem, AlertEngine, ForecastPerformanceTracker

def test_transparent_scoring_system():
    scorer = TransparentScoringSystem()
    scores = scorer.compute_scores(
        rsi_14=65.0,
        dist_sma_50=0.03,
        macd_hist=0.5,
        macro_regime="Inflationary",
        volatility_20d=0.18,
        gs_ratio_zscore=1.2,
        analogue_pos_prob=0.64
    )
    assert "composite_score" in scores
    assert 0 <= scores["composite_score"] <= 100
    assert scores["macro_score"] == 75.0

def test_alert_engine(tmp_path):
    gold_df = pd.DataFrame({"close": [2000.0, 2080.0]}) # 4% jump
    silver_df = pd.DataFrame({"close": [25.0, 25.1]})

    alerts = AlertEngine.detect_alerts("2024-01-01", gold_df, silver_df, {}, output_dir=str(tmp_path))
    assert len(alerts) >= 1
    assert alerts[0]["type"] == "GOLD_LARGE_MOVE"
    assert (tmp_path / "2024-01-01.json").exists()

def test_forecast_performance_tracker(tmp_path):
    tracker = ForecastPerformanceTracker(storage_path=str(tmp_path / "model-performance.json"))
    history = [
        {"predicted_direction": 1, "actual_return": 0.02}, # correct
        {"predicted_direction": 1, "actual_return": -0.01}, # wrong
        {"predicted_direction": 0, "actual_return": -0.02} # correct
    ]
    res = tracker.update_performance(history)
    assert res["evaluated_samples"] == 3
    assert abs(res["overall_directional_accuracy"] - (2/3)) < 1e-4
