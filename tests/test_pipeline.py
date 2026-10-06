import pytest
import pandas as pd
import numpy as np

from src.ingestion.pipeline import DataIngestionPipeline
from src.ingestion.yfinance_provider import YFinanceProvider
from src.validation.validator import DataValidator
from src.indicators.technical import TechnicalIndicators
from src.ratios.gold_silver import GoldSilverRatioAnalyzer
from src.macro.correlations import MacroCorrelationAnalyzer
from src.regimes.detector import MacroRegimeDetector, MarketRegimeDetector
from src.patterns.analogue_engine import HistoricalAnalogueEngine
from src.forecasting.probabilistic import ForwardProbabilityEngine
from src.features.generator import DailyFeatureStoreGenerator
from src.models.baseline import ForecastingModels
from src.evaluation.evaluator import ForecastTracker, ForecastEvaluator
from src.backtesting.engine import BacktestingEngine
from src.scoring.system import TransparentScoringSystem
from src.reporting.alerts import AlertDetector
from src.reporting.generator import ReportGenerator

@pytest.fixture
def mock_prices():
    dates = pd.date_range(start="2026-01-01", periods=50, freq="B").strftime("%Y-%m-%d")
    np.random.seed(42)
    p_gold = 2000.0 + np.cumsum(np.random.normal(1.0, 10.0, 50))
    p_silver = 24.0 + np.cumsum(np.random.normal(0.05, 0.5, 50))

    df_gold = pd.DataFrame({
        "date": dates,
        "open": p_gold * 0.99,
        "high": p_gold * 1.01,
        "low": p_gold * 0.98,
        "close": p_gold,
        "volume": np.random.randint(1000, 5000, 50)
    })
    df_silver = pd.DataFrame({
        "date": dates,
        "open": p_silver * 0.99,
        "high": p_silver * 1.01,
        "low": p_silver * 0.98,
        "close": p_silver,
        "volume": np.random.randint(2000, 10000, 50)
    })
    return df_gold, df_silver

def test_data_ingestion_and_currency_conversion(mock_prices):
    provider = YFinanceProvider()
    pipeline = DataIngestionPipeline(provider=provider)
    data = pipeline.fetch_all("2026-01-01", "2026-02-15")

    assert "gold_usd" in data
    assert "gold_inr" in data
    assert "silver_inr" in data

    gold_inr = data["gold_inr"]
    assert not gold_inr.empty
    assert "exchange_rate_used" in gold_inr.columns
    assert gold_inr["currency"].iloc[0] == "INR"
    assert gold_inr["unit"].iloc[0] == "10g"

def test_data_validation(mock_prices):
    df_gold, _ = mock_prices
    validator = DataValidator()
    v_df, summary = validator.validate_dataset(df_gold, "GOLD")

    assert summary["records"] == 50
    assert summary["invalid_records"] == 0
    assert "validation_status" in v_df.columns

def test_technical_indicators(mock_prices):
    df_gold, _ = mock_prices
    tech = TechnicalIndicators.calculate_all(df_gold)

    assert "sma_20" in tech.columns
    assert "ema_9" in tech.columns
    assert "rsi_14" in tech.columns
    assert "macd" in tech.columns
    assert "atr_14" in tech.columns
    assert "volatility_20d" in tech.columns
    assert not tech["rsi_14"].isnull().all()

def test_gold_silver_ratio(mock_prices):
    df_gold, df_silver = mock_prices
    ratio_df = GoldSilverRatioAnalyzer.analyze_ratio(df_gold, df_silver)

    assert "gold_silver_ratio" in ratio_df.columns
    assert "ratio_zscore_252d" in ratio_df.columns
    summary = GoldSilverRatioAnalyzer.get_latest_summary(ratio_df)
    assert summary["current_ratio"] > 0

def test_macro_correlations(mock_prices):
    df_gold, df_silver = mock_prices
    corrs = MacroCorrelationAnalyzer.compute_correlations(df_gold, {"silver": df_silver})

    assert "corr_silver_20d" in corrs.columns

def test_regime_detection(mock_prices):
    df_gold, _ = mock_prices
    macro_regime = MacroRegimeDetector.classify_regime({}, "2026-02-01")
    assert macro_regime["primary_regime"] in ["Neutral", "Risk-On", "Risk-Off", "Inflationary", "Disinflationary", "Tightening", "Easing", "Stagflationary"]

    mkt_regime = MarketRegimeDetector.classify_market_regime(TechnicalIndicators.calculate_all(df_gold))
    assert "trend_regime" in mkt_regime
    assert "volatility_regime" in mkt_regime

def test_feature_generator_and_analogue_engine(mock_prices):
    df_gold, df_silver = mock_prices
    tech_gold = TechnicalIndicators.calculate_all(df_gold)
    feats = DailyFeatureStoreGenerator.generate_features(tech_gold, {}, "GOLD")

    assert "future_return_5d" in feats.columns
    assert "days_since_event" in feats.columns

    target_date = feats["date"].iloc[40]
    analogues = HistoricalAnalogueEngine.find_analogues(feats, target_date, top_k=5)
    assert "sample_count" in analogues

    fwd = ForwardProbabilityEngine.calculate_forward_probabilities(feats, target_date, "GOLD")
    assert "disclaimer" in fwd

def test_models_and_backtesting(mock_prices):
    df_gold, _ = mock_prices
    tech_gold = TechnicalIndicators.calculate_all(df_gold)
    feats = DailyFeatureStoreGenerator.generate_features(tech_gold, {}, "GOLD")

    target_date = feats["date"].iloc[45]
    models = ForecastingModels()
    preds = models.train_and_predict(feats, target_date, horizon=5)
    assert "predicted_direction" in preds

    feats["signal"] = 1
    bt = BacktestingEngine()
    bt_res = bt.run_backtest(feats, "signal")
    assert "cumulative_return" in bt_res

def test_scoring_and_reporting(mock_prices):
    scorer = TransparentScoringSystem()
    scores = scorer.compute_scores(
        {"dist_sma_200": 0.05, "rsi_14": 65, "volatility_20d": 0.12},
        {"primary_regime": "Inflationary"},
        {"ratio_zscore": 1.2},
        {"forward_stats": {"5d": {"prob_positive": 0.62}}},
        {"model_agreement": "3/4"}
    )
    assert 0 <= scores["composite_score"] <= 100
    assert 0 <= scores["confidence_score"] <= 100

    alerts = AlertDetector().detect_alerts("2026-02-01", {"rsi_14": 75}, {"ratio_zscore": 2.5})
    assert alerts["alerts_triggered"] == 2

    md = ReportGenerator.generate_markdown_report("2026-02-01", {"price": 2050}, {"price": 25}, {"current_ratio": 82}, {"primary_regime": "Inflationary"}, {}, {}, {})
    assert "# Precious Metals Daily Research Report" in md
