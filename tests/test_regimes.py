import pytest
from src.regimes.macro_regime import MacroRegimeDetector
from src.regimes.market_regime import MarketRegimeDetector
from src.ingestion.collector import DataCollector
from src.indicators.technical import TechnicalIndicators


def test_regime_detectors():
    collector = DataCollector(use_fallback=True)
    data = collector.collect_all("2025-01-01", "2026-01-15")

    macro_reg = MacroRegimeDetector.detect_regime(data)
    assert "primary_regime" in macro_reg
    assert "explanation" in macro_reg

    gold_ti = TechnicalIndicators.calculate_all(data["GOLD"])
    mkt_reg = MarketRegimeDetector.detect_market_regime(gold_ti, "GOLD")
    assert "trend_regime" in mkt_reg
    assert "volatility_regime" in mkt_reg
