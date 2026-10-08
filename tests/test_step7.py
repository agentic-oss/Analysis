import pytest
from src.reporting.daily_report import DailyReportGenerator
from src.reporting.alerts import AlertEngine

def test_daily_report_and_alerts(tmp_path):
    date_str = "2026-03-30"
    gold_analysis = {"price_usd_oz": 2500.0, "rsi_14": 75.0, "golden_cross": True}
    silver_analysis = {"price_usd_oz": 30.0, "rsi_14": 40.0}
    ratio_analysis = {"current_ratio": 83.3, "zscore": 2.2}
    macro_regime = {"primary_regime": "Risk-Off", "description": "High volatility"}
    market_regimes = {"GOLD": {"composite_regime": "Strong Bullish Trend | Normal Volatility"}}
    gold_forecast = {"research_bias": "Bullish", "horizons": {"5d": {"probability_positive_return_pct": 65.0}}}
    silver_forecast = {"research_bias": "Neutral", "horizons": {"5d": {"probability_positive_return_pct": 50.0}}}
    quality_summary = {"total_valid_records": 100, "total_suspicious_records": 0, "errors": []}

    report_path = DailyReportGenerator.generate_markdown_report(
        date_str, gold_analysis, silver_analysis, ratio_analysis, macro_regime,
        market_regimes, gold_forecast, silver_forecast, quality_summary, output_dir=str(tmp_path)
    )
    assert "2026-03-30.md" in report_path

    alerts = AlertEngine.detect_alerts(date_str, gold_analysis, silver_analysis, ratio_analysis, macro_regime)
    assert alerts["total_alerts"] > 0
