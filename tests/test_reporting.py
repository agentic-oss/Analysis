import pytest
from src.reporting.report_generator import DailyReportGenerator
from src.pipeline.runner import run_pipeline


def test_report_generator():
    gold_analysis = {"price": 2500.0, "daily_move_pct": 1.2, "trend": "Bullish", "research_bias": "Bullish Bias"}
    silver_analysis = {"price": 30.0, "daily_move_pct": 0.5, "trend": "Bullish", "research_bias": "Bullish Bias"}

    md = DailyReportGenerator.generate_markdown_report(
        "2026-09-08", gold_analysis, silver_analysis, {}, {}, {}, {}, [], {}, {}, {}, {}, {}
    )
    assert "# Precious Metals Daily Research Report — 2026-09-08" in md
    assert "Gold Analysis" in md
    assert "Silver Analysis" in md


def test_end_to_end_pipeline():
    res = run_pipeline("2026-09-08")
    assert res["date"] == "2026-09-08"
    assert "gold" in res
    assert "silver" in res
    assert "scores" in res
