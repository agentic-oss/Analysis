import pytest
import os
import json
import pandas as pd
from src.evaluation.tracker import ForecastPerformanceTracker
from src.scoring.composite import ScoringSystem
from src.scoring.alerts import AlertDetector
from src.reporting.generator import ReportGenerator

def test_scoring_system():
    scoring = ScoringSystem()
    scores = scoring.calculate_scores(
        rsi_14=65.0,
        dist_sma_20=1.5,
        dist_sma_200=5.0,
        macd_hist=0.5,
        macro_regime="Inflationary",
        gsr_zscore=1.8,
        analogue_pos_prob=64.0,
        hist_vol_20=0.15
    )
    assert "composite_score" in scores
    assert 0.0 <= scores["composite_score"] <= 100.0

def test_alert_detector(tmp_path):
    detector = AlertDetector(alerts_dir=str(tmp_path))
    res = detector.detect_alerts(
        date_str="2026-03-01",
        gold_data={"rsi_14": 75.0},
        silver_data={"rsi_14": 25.0},
        gsr_data={"z_score": 2.2},
        macro_data={"regime": "Stagflationary"}
    )
    assert res["total_alerts"] == 4
    assert os.path.exists(os.path.join(str(tmp_path), "2026-03-01.json"))

def test_report_generator(tmp_path):
    rep_dir = os.path.join(tmp_path, "reports")
    json_dir = os.path.join(tmp_path, "json")
    generator = ReportGenerator(reports_dir=rep_dir, analysis_json_dir=json_dir)

    payload = {
        "gold": {"price": 2000.0, "return_1d_pct": 0.5},
        "silver": {"price": 25.0, "return_1d_pct": 1.0},
        "macro": {"regime": "Inflationary"},
        "gold_silver_ratio": {"current_ratio": 80.0, "z_score": 0.5},
        "historical_analogues": {"top_k": 2, "analogues": []},
        "forecasts": {},
        "data_quality": {"invalid_records": 0}
    }

    md_path = generator.generate_markdown_report("2026-03-01", payload)
    json_path = generator.generate_json_report("2026-03-01", payload)

    assert os.path.exists(md_path)
    assert os.path.exists(json_path)
