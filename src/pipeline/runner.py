import sys
import os
import logging
from datetime import datetime, timedelta
import yaml
import json
import pandas as pd

from src.ingestion.provider import YFinanceDataProvider
from src.ingestion.indian_market import process_indian_price_dataframe, convert_usd_oz_to_inr_10g, convert_usd_oz_to_inr_kg
from src.validation.validator import DataValidator
from src.storage.storage_manager import StorageManager
from src.indicators.technical import TechnicalIndicators
from src.macro.macro_analyzer import MacroAnalyzer
from src.metals.gold_analyzer import GoldAnalyzer
from src.metals.silver_analyzer import SilverAnalyzer
from src.ratios.gold_silver_ratio import GoldSilverRatioAnalyzer
from src.regimes.macro_regimes import MacroRegimeModel
from src.regimes.market_regimes import AssetMarketRegimeModel
from src.patterns.historical_analogues import HistoricalAnalogueEngine
from src.features.feature_builder import FeatureBuilder
from src.scoring.score_calculator import ScoreCalculator
from src.scoring.confidence import PredictionConfidenceFramework
from src.scoring.alerts import AlertDetector
from src.forecasting.forecast_engine import ProbabilisticForecastEngine
from src.models.baseline_models import BaselineModels
from src.evaluation.forecast_tracker import ForecastTracker
from src.reporting.report_generator import DailyReportGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PipelineRunner")


def run_pipeline(date_str: str = None):
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    logger.info(f"Starting Precious Metals Daily Pipeline for date: {date_str}")

    # Load system config
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_yaml_load(f) if hasattr(yaml, "safe_yaml_load") else yaml.safe_load(f)

    storage = StorageManager(base_data_dir="data")
    provider = YFinanceDataProvider()
    validator = DataValidator()

    # Define historical window (up to 2 years lookback for robust analytics)
    start_dt = (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=730)).strftime("%Y-%m-%d")
    end_dt = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

    # 1. Fetch market data
    logger.info("Ingesting market data...")
    gold_df = provider.get_historical_prices("GC=F", start_dt, end_dt)
    silver_df = provider.get_historical_prices("SI=F", start_dt, end_dt)
    usdinr_df = provider.get_currency_prices("INR=X", start_dt, end_dt)
    dxy_df = provider.get_currency_prices("DX-Y.NYB", start_dt, end_dt)
    us10y_df = provider.get_macro_data("^TNX", start_dt, end_dt)
    us2y_df = provider.get_macro_data("^IRX", start_dt, end_dt)
    oil_df = provider.get_macro_data("CL=F", start_dt, end_dt)
    sp500_df = provider.get_macro_data("^GSPC", start_dt, end_dt)
    nifty_df = provider.get_macro_data("^NSEI", start_dt, end_dt)
    vix_df = provider.get_macro_data("^VIX", start_dt, end_dt)
    indiavix_df = provider.get_macro_data("^INDIAVIX", start_dt, end_dt)
    events_df = provider.get_economic_indicators()

    # 2. Data Validation & Quality Summary
    logger.info("Validating ingestion batches...")
    val_reports = []
    gold_df, r1 = validator.validate_dataset(gold_df, "GOLD")
    silver_df, r2 = validator.validate_dataset(silver_df, "SILVER")
    val_reports.extend([r1, r2])

    quality_summary = validator.create_daily_quality_report(date_str, val_reports)
    quality_file = os.path.join(config["paths"]["quality"], f"{date_str}.json")
    storage.save_json(quality_summary, quality_file)

    # 3. Process Indian Market Converted Prices
    logger.info("Processing Indian market price conversions...")
    gold_df, silver_df = process_indian_price_dataframe(gold_df, silver_df, usdinr_df)

    # 4. Save Processed Historical Datasets
    storage.append_or_update_df(gold_df, os.path.join(config["paths"]["processed_metals"], "gold_daily"), dedup_cols=["Date"])
    storage.append_or_update_df(silver_df, os.path.join(config["paths"]["processed_metals"], "silver_daily"), dedup_cols=["Date"])

    # 5. Technical Indicators
    logger.info("Calculating technical indicators...")
    gold_df = TechnicalIndicators.calculate_all(gold_df)
    silver_df = TechnicalIndicators.calculate_all(silver_df)

    # 6. Macro Environment Analysis
    logger.info("Analyzing macro drivers and cross-asset relationships...")
    macro_analyzer = MacroAnalyzer()
    macro_summary = macro_analyzer.analyze_macro_environment(
        dxy_df, us10y_df, us2y_df, oil_df, sp500_df, nifty_df, vix_df, indiavix_df, usdinr_df
    )

    # 7. Gold/Silver Ratio & Relative Value
    logger.info("Analyzing Gold/Silver ratio...")
    ratio_df = GoldSilverRatioAnalyzer.calculate_ratio_series(gold_df, silver_df)
    ratio_summary = GoldSilverRatioAnalyzer.analyze_latest_ratio(ratio_df)

    # 8. Regimes & Cross Correlations
    logger.info("Detecting macro and asset market regimes...")
    macro_regime = MacroRegimeModel.classify_macro_regime(macro_summary)
    gold_market_regime = AssetMarketRegimeModel.classify_market_regime(gold_df)
    silver_market_regime = AssetMarketRegimeModel.classify_market_regime(silver_df)

    macro_dfs = {"dxy": dxy_df, "us10y": us10y_df, "oil": oil_df, "sp500": sp500_df, "vix": vix_df}
    gold_corrs = GoldAnalyzer.calculate_rolling_correlations(gold_df, macro_dfs)
    silver_corrs = SilverAnalyzer.calculate_rolling_correlations(silver_df, macro_dfs)

    gold_supp_res = GoldAnalyzer.find_support_resistance(gold_df)
    silver_supp_res = SilverAnalyzer.find_support_resistance(silver_df)

    # 9. Historical Analogues & Forward Statistics
    logger.info("Searching historical market state analogues...")
    analogue_engine = HistoricalAnalogueEngine(top_k=15)
    gold_analogues = analogue_engine.find_analogues(gold_df, date_str)
    silver_analogues = analogue_engine.find_analogues(silver_df, date_str)

    # 10. Scoring & Prediction Confidence
    logger.info("Calculating composite research scores and prediction confidence...")
    gold_scores = ScoreCalculator.calculate_composite_score(gold_df, macro_summary, ratio_summary)
    silver_scores = ScoreCalculator.calculate_composite_score(silver_df, macro_summary, ratio_summary)

    gold_confidence = PredictionConfidenceFramework.evaluate_confidence(gold_scores, gold_analogues, gold_market_regime)
    silver_confidence = PredictionConfidenceFramework.evaluate_confidence(silver_scores, silver_analogues, silver_market_regime)

    # 11. Alert Detection
    logger.info("Checking for significant market alerts...")
    alerts_summary = AlertDetector.detect_alerts(gold_df, silver_df, macro_summary, ratio_summary, date_str)
    alerts_file = os.path.join(config["paths"]["analysis_alerts"], f"{date_str}.json")
    storage.save_json(alerts_summary, alerts_file)

    # 12. Probabilistic Forecasts
    logger.info("Generating forward research forecast probabilities...")
    gold_forecasts = ProbabilisticForecastEngine.generate_forecasts(gold_df, gold_analogues)
    silver_forecasts = ProbabilisticForecastEngine.generate_forecasts(silver_df, silver_analogues)

    # 13. ML Feature Store Builder
    logger.info("Building daily ML feature datasets...")
    gold_features = FeatureBuilder.build_feature_dataset(gold_df, "GOLD", macro_summary, ratio_df, events_df)
    silver_features = FeatureBuilder.build_feature_dataset(silver_df, "SILVER", macro_summary, ratio_df, events_df)
    storage.save_df(gold_features, os.path.join(config["paths"]["features"], f"{date_str}_gold_features"))

    # 14. Forecast Recording & Performance Tracking
    logger.info("Recording forecast and updating evaluation performance database...")
    tracker = ForecastTracker(predictions_dir=config["paths"]["predictions_daily"], performance_file=os.path.join(config["paths"]["models"], "model-performance.json"))
    g_latest_price = float(gold_df["Close"].iloc[-1]) if not gold_df.empty else 0.0
    s_latest_price = float(silver_df["Close"].iloc[-1]) if not silver_df.empty else 0.0

    tracker.record_forecast(date_str, "GOLD", gold_forecasts, gold_confidence, g_latest_price)
    tracker.record_forecast(date_str, "SILVER", silver_forecasts, silver_confidence, s_latest_price)
    tracker.evaluate_historical_forecasts(gold_df, "GOLD")

    # 15. Reporting Output Generation
    logger.info("Generating daily JSON analysis and Markdown research report...")
    latest_g_inr = float(gold_df["Close_INR_10g"].iloc[-1]) if "Close_INR_10g" in gold_df.columns else convert_usd_oz_to_inr_10g(g_latest_price, 83.0)
    latest_s_inr = float(silver_df["Close_INR_kg"].iloc[-1]) if "Close_INR_kg" in silver_df.columns else convert_usd_oz_to_inr_kg(s_latest_price, 83.0)

    gold_analysis_summary = {
        "price": round(g_latest_price, 2),
        "price_inr_10g": round(latest_g_inr, 2),
        "daily_move_pct": round(float(gold_df["return_1d"].iloc[-1] * 100.0) if "return_1d" in gold_df.columns else 0.0, 2),
        "rsi": round(float(gold_df["rsi_14"].iloc[-1]) if "rsi_14" in gold_df.columns else 50.0, 1),
        "dist_200dma": round(float(gold_df["distance_sma_200"].iloc[-1] * 100.0) if "distance_sma_200" in gold_df.columns else 0.0, 2),
        "regime": gold_market_regime,
        "correlations": gold_corrs,
        "support": gold_supp_res.get("support", []),
        "resistance": gold_supp_res.get("resistance", []),
        "scores": gold_scores,
        "confidence": gold_confidence,
        "forward_statistics": gold_forecasts.get("forecasts_by_horizon", {}),
    }

    silver_analysis_summary = {
        "price": round(s_latest_price, 2),
        "price_inr_kg": round(latest_s_inr, 2),
        "daily_move_pct": round(float(silver_df["return_1d"].iloc[-1] * 100.0) if "return_1d" in silver_df.columns else 0.0, 2),
        "rsi": round(float(silver_df["rsi_14"].iloc[-1]) if "rsi_14" in silver_df.columns else 50.0, 1),
        "dist_200dma": round(float(silver_df["distance_sma_200"].iloc[-1] * 100.0) if "distance_sma_200" in silver_df.columns else 0.0, 2),
        "regime": silver_market_regime,
        "correlations": silver_corrs,
        "support": silver_supp_res.get("support", []),
        "resistance": silver_supp_res.get("resistance", []),
        "scores": silver_scores,
        "confidence": silver_confidence,
        "forward_statistics": silver_forecasts.get("forecasts_by_horizon", {}),
    }

    market_regimes = {
        "primary_macro_regime": macro_regime.get("primary_regime"),
        "macro_components": macro_regime.get("components"),
        "gold_regime": gold_market_regime,
        "silver_regime": silver_market_regime,
    }

    daily_json_file = os.path.join(config["paths"]["analysis_daily"], f"{date_str}.json")
    json_output = DailyReportGenerator.generate_json_output(
        date_str,
        gold_analysis_summary,
        silver_analysis_summary,
        ratio_summary,
        macro_summary,
        market_regimes,
        quality_summary,
        alerts_summary,
        daily_json_file,
    )

    daily_report_file = os.path.join(config["paths"]["reports_daily"], f"{date_str}.md")
    DailyReportGenerator.generate_markdown_report(date_str, json_output, daily_report_file)

    logger.info(f"Pipeline execution completed successfully for {date_str}!")


if __name__ == "__main__":
    target_date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    run_pipeline(target_date)
