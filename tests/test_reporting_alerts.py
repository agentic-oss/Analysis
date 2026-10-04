import pytest
import os
import json
from src.reporting.daily_report import AlertEngine, ResearchReportGenerator

def test_alert_engine():
    gold_row = {"close": 2000.0, "return_1d": 0.04, "rsi_14": 75.0, "breakout_20d": True}
    ratio_row = {"ratio_zscore": 2.2}
    macro_dict = {"explainable_components": {}}
    config = {"alerts": {"price_jump_pct": 3.5, "rsi_overbought": 70, "ratio_zscore_extreme": 2.0}}

    alerts = AlertEngine.detect_alerts("2025-01-15", gold_row, None, ratio_row, macro_dict, config)
    assert alerts["alert_count"] >= 3
    types = [a["type"] for a in alerts["alerts"]]
    assert "LARGE_PRICE_MOVE" in types
    assert "EXTREME_RSI" in types
    assert "BREAKOUT" in types

def test_report_generator(tmp_path):
    analysis_payload = {
        "date": "2025-01-15",
        "gold": {"price": 2000.0, "return_1d_pct": 0.5, "signals": {}},
        "silver": {"price": 25.0, "return_1d_pct": 0.2, "signals": {}},
        "macro": {},
        "gold_silver_ratio": {},
        "data_quality": {}
    }
    path = ResearchReportGenerator.generate_markdown_report("2025-01-15", analysis_payload, output_dir=str(tmp_path))
    assert os.path.exists(path)
    with open(path) as f:
        content = f.read()
    assert "# Precious Metals Daily Research Report" in content
