import os
import pytest
from src.pipeline.runner import run_pipeline

def test_pipeline_execution():
    test_date = "2026-03-31"
    run_pipeline(execution_date=test_date, use_synthetic=True)

    assert os.path.exists(f"data/quality/{test_date}.json")
    assert os.path.exists(f"data/analysis/daily/{test_date}.json")
    assert os.path.exists(f"data/analysis/alerts/{test_date}.json")
    assert os.path.exists(f"reports/daily/{test_date}.md")
