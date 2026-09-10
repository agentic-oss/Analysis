import pytest
from src.reporting.generator import DailyReportGenerator

def test_daily_report_generator(tmp_path):
    generator = DailyReportGenerator()
    g_sum = {"price": 2000.0, "daily_move_pct": 0.5, "technical_bias": "Bullish", "rsi_14": 60.0, "volatility_20d": 0.15}
    s_sum = {"price": 25.0, "daily_move_pct": 1.0, "technical_bias": "Bullish", "rsi_14": 62.0, "volatility_20d": 0.20}
    m_sum = {"primary_macro_regime": "Inflationary", "inputs": {}}
    rv_sum = {"gold_silver_ratio": 80.0, "ratio_zscore_60": 0.5}
    an_sum = {"sample_count": 25}
    fc_sum = {"disclaimer": "Historical statistics.", "signals": {"5d": {"bias": "Bullish", "positive_return_probability": 0.65, "expected_return": 0.015}}}
    q_sum = {"overall_status": "valid", "instruments_evaluated": 10, "all_warnings": [], "all_errors": []}
    sc_sum = {"composite_score": 72.5}

    md_path = generator.generate_markdown_report(
        "2024-01-01", g_sum, s_sum, m_sum, rv_sum, an_sum, fc_sum, q_sum, output_dir=str(tmp_path)
    )
    assert (tmp_path / "2024-01-01.md").exists()

    json_path = generator.generate_json_report(
        "2024-01-01", g_sum, s_sum, m_sum, rv_sum, sc_sum, fc_sum, q_sum, output_dir=str(tmp_path)
    )
    assert (tmp_path / "2024-01-01.json").exists()
