import os
import pytest
import json
import pandas as pd
from src.reporting.generator import ReportGenerator
from src.pipeline.runner import run_pipeline

def test_report_generator(tmp_path):
    reporter = ReportGenerator(
        reports_dir=str(tmp_path / "reports"),
        analysis_dir=str(tmp_path / "analysis"),
        alerts_dir=str(tmp_path / "alerts")
    )

    payload = {
        "date": "2025-01-01",
        "gold": {"price": 2000.0, "daily_return_pct": 0.5, "composite_score": 65.0, "forward_statistics": {}},
        "silver": {"price": 25.0, "daily_return_pct": -2.5, "composite_score": 55.0, "forward_statistics": {}},
        "gold_silver_ratio": {"current_ratio": 80.0, "current_zscore": 2.2},
        "macro": {"primary_macro_regime": "Risk-Off"},
        "market_regime": {"gold": {"trend_regime": "Bullish"}, "silver": {"trend_regime": "Range"}},
        "data_quality": {"valid_records": 100, "suspicious_records": 0, "invalid_records": 0}
    }

    md_file = reporter.generate_daily_markdown_report("2025-01-01", payload)
    json_file = reporter.generate_daily_json_output("2025-01-01", payload)
    alert_file = reporter.detect_and_save_alerts("2025-01-01", payload, {"price_change_pct_1d": 2.0, "gsr_zscore_extreme": 2.0})

    assert os.path.exists(md_file)
    assert os.path.exists(json_file)
    assert os.path.exists(alert_file)

    with open(alert_file, "r") as f:
        alerts = json.load(f)
        assert alerts["alerts_count"] >= 1

def test_pipeline_runner():
    target_date = "2025-01-15"
    run_pipeline(target_date, use_synthetic=True)

    assert os.path.exists(f"reports/daily/{target_date}.md")
    assert os.path.exists(f"data/analysis/daily/{target_date}.json")
    assert os.path.exists(f"data/analysis/alerts/{target_date}.json")
