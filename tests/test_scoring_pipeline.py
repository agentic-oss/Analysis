"""
Unit tests for scoring, alert detection, reporting, and pipeline orchestration.
"""

import os
import pytest
from src.scoring.scoring import ScoringEngine, AlertEngine
from src.reporting.report_generator import ReportGenerator
from src.pipeline.runner import DailyPipelineRunner


def test_scoring_engine():
    engine = ScoringEngine()
    scores = engine.compute_composite_scores(
        tech_indicators={"rsi_14": 60.0, "dist_sma_200_pct": 5.0, "volatility_20d": 0.12},
        macro_indicators={"real_yield_change_20d": -0.05, "dxy_change_20d": -0.01},
        ratio_metrics={"ratio_zscore": 0.5},
        analogue_stats={"5d": {"prob_positive": 0.65}},
    )
    assert 0 <= scores["composite_score"] <= 100
    assert "technical_score" in scores


def test_report_generator():
    gen = ReportGenerator()
    data = {
        "gold": {"price_usd": 2000.0, "return_1d_pct": 0.5},
        "silver": {"price_usd": 25.0, "return_1d_pct": 1.0},
        "ratio": {"gold_silver_ratio": 80.0},
        "macro": {"primary_regime": "Inflationary"},
        "scores": {"composite_score": 65.0},
        "alerts": [],
        "data_quality": {"valid_records": 10},
    }
    md = gen.generate_markdown_report("2026-03-31", data)
    assert "# Precious Metals Daily Research Report — 2026-03-31" in md
    assert "Gold Summary" in md


def test_pipeline_runner(tmp_path):
    runner = DailyPipelineRunner()
    result = runner.run_pipeline("2026-03-31")
    assert "gold" in result
    assert "silver" in result
    assert os.path.exists("reports/daily/2026-03-31.md")
    assert os.path.exists("data/analysis/daily/2026-03-31.json")
