"""Daily Precious Metals Research Pipeline Orchestrator."""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, Any, List
import pandas as pd

from src.ingestion.provider import YahooFinanceProvider
from src.validation.validator import DataValidator
from src.storage.manager import StorageManager
from src.indicators.technical import TechnicalIndicators
from src.metals.analyzer import MetalAnalyzer
from src.ratios.relative_value import RelativeValueAnalyzer
from src.macro.regime import MacroRegimeClassifier
from src.regimes.market import MarketRegimeClassifier
from src.patterns.analogue import HistoricalAnalogueEngine
from src.forecasting.engine import ForwardProbabilityEngine
from src.features.generator import FeatureStoreGenerator
from src.models.forecaster import MetalsForecastingModels
from src.backtesting.engine import BacktestEngine
from src.evaluation.tracker import ForecastEvaluator
from src.scoring.composite import CompositeScorer
from src.scoring.alerts import AlertDetector
from src.reporting.generator import ReportGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PipelineRunner")


class DailyMetalsPipeline:
    """Orchestrates end-to-end precious metals data ingestion, analysis, forecasting, report generation, and tracking."""

    def __init__(self, data_dir: str = "data", reports_dir: str = "reports"):
        self.provider = YahooFinanceProvider()
        self.validator = DataValidator()
        self.storage = StorageManager(base_dir=data_dir)
        self.data_dir = data_dir
        self.reports_dir = reports_dir

    def run_pipeline(self, target_date_str: str) -> Dict[str, Any]:
        """Runs the complete daily execution loop for a given date string (YYYY-MM-DD)."""
        logger.info(f"=== STARTING DAILY PRECIOUS METALS PIPELINE FOR {target_date_str} ===")

        start_date = (datetime.strptime(target_date_str, "%Y-%m-%d") - timedelta(days=365*2)).strftime("%Y-%m-%d")
        end_date = (datetime.strptime(target_date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

        # 1. Ingestion
        logger.info("1. Ingesting market and macro data...")
        gold_raw = self.provider.get_spot_prices("GOLD", start_date, end_date)
        silver_raw = self.provider.get_spot_prices("SILVER", start_date, end_date)
        gold_inr_raw = self.provider.get_spot_prices("GOLD_INR", start_date, end_date)
        silver_inr_raw = self.provider.get_spot_prices("SILVER_INR", start_date, end_date)
        usdinr_raw = self.provider.get_currency_prices("USDINR", start_date, end_date)

        dxy_raw = self.provider.get_macro_data("DXY", start_date, end_date)
        us10y_raw = self.provider.get_macro_data("US10Y", start_date, end_date)
        sp500_raw = self.provider.get_macro_data("SP500", start_date, end_date)
        vix_raw = self.provider.get_macro_data("VIX", start_date, end_date)
        oil_raw = self.provider.get_macro_data("CRUDE_OIL", start_date, end_date)

        # 2. Validation & Quality Reporting
        logger.info("2. Validating ingested records...")
        gold_val, q_gold = self.validator.validate_dataframe(gold_raw, "GOLD")
        silver_val, q_silver = self.validator.validate_dataframe(silver_raw, "SILVER")

        quality_report = {
            "date": target_date_str,
            "gold_records": q_gold["total_records"],
            "silver_records": q_silver["total_records"],
            "warnings": q_gold["warnings"] + q_silver["warnings"],
            "errors": q_gold["errors"] + q_silver["errors"]
        }
        self.validator.save_quality_report(target_date_str, quality_report)

        # 3. Technical Indicators
        logger.info("3. Calculating technical indicators...")
        gold_df = TechnicalIndicators.calculate_all(gold_val)
        silver_df = TechnicalIndicators.calculate_all(silver_val)

        # Persist Processed Parquet Data
        self.storage.save_parquet(gold_df, "processed/metals/gold.parquet", key_cols=['date'])
        self.storage.save_parquet(silver_df, "processed/metals/silver.parquet", key_cols=['date'])

        # 4. Metal Structural Summaries & Correlations
        logger.info("4. Analyzing Gold and Silver structure and macro correlations...")
        metal_analyzer = MetalAnalyzer()
        gold_summary = metal_analyzer.analyze_metal_structure(gold_df)
        silver_summary = metal_analyzer.analyze_metal_structure(silver_df)

        macro_dfs = {
            "DXY": dxy_raw,
            "US10Y": us10y_raw,
            "SP500": sp500_raw,
            "VIX": vix_raw,
            "CRUDE_OIL": oil_raw
        }
        gold_corrs = metal_analyzer.calculate_correlations(gold_df, macro_dfs)

        # 5. Ratio & Relative Value
        logger.info("5. Calculating Gold/Silver ratio and relative value...")
        ratio_df = RelativeValueAnalyzer.calculate_ratio(gold_df, silver_df)
        if not ratio_df.empty:
            self.storage.save_parquet(ratio_df, "processed/metals/gold_silver_ratio.parquet", key_cols=['date'])
        ratio_summary = RelativeValueAnalyzer.analyze_ratio_extremes(ratio_df)

        # 6. Macro & Market Regimes
        logger.info("6. Detecting Macro and Market Regimes...")
        macro_summary = MacroRegimeClassifier.classify(macro_dfs)
        gold_market_regime = MarketRegimeClassifier.classify_market(gold_df)
        silver_market_regime = MarketRegimeClassifier.classify_market(silver_df)

        gold_summary["market_regime"] = gold_market_regime
        silver_summary["market_regime"] = silver_market_regime

        # 7. Feature Store Generation
        logger.info("7. Generating ML Feature Store...")
        events = self.provider.get_event_data(start_date, end_date)
        gold_features = FeatureStoreGenerator.build_feature_dataset("GOLD", gold_df, ratio_df, macro_dfs, events)
        if not gold_features.empty:
            self.storage.save_parquet(gold_features, "features/daily/gold_features.parquet", key_cols=['date'])

        # 8. Historical Analogue Search & Forward Probabilities
        logger.info("8. Searching historical analogues & generating forward research signals...")
        analogue_engine = HistoricalAnalogueEngine()
        gold_analogues = analogue_engine.find_analogues(gold_features, target_date_str) if not gold_features.empty else {}
        silver_analogues = analogue_engine.find_analogues(gold_features, target_date_str) if not gold_features.empty else {}

        gold_forecasts = ForwardProbabilityEngine.generate_probabilistic_signal(
            gold_analogues.get("forward_stats", {}), gold_summary, macro_summary
        )
        silver_forecasts = ForwardProbabilityEngine.generate_probabilistic_signal(
            silver_analogues.get("forward_stats", {}), silver_summary, macro_summary
        )

        # 9. Scoring & Alerts
        logger.info("9. Calculating composite scores and evaluating alerts...")
        gold_scores = CompositeScorer.calculate_scores(gold_summary, macro_summary, ratio_summary, gold_analogues)
        silver_scores = CompositeScorer.calculate_scores(silver_summary, macro_summary, ratio_summary, silver_analogues)

        alerts = AlertDetector.detect_alerts(target_date_str, gold_summary, silver_summary, ratio_summary, macro_summary)
        AlertDetector.save_alerts(target_date_str, alerts)

        # 10. Forecast Performance Evaluation Tracking
        logger.info("10. Evaluating previous forecasts against actual outcomes...")
        evaluator = ForecastEvaluator()
        saved_forecasts_path = os.path.join(self.data_dir, "predictions/daily/history.json")
        saved_forecasts = self.storage.read_json("predictions/daily/history.json").get("records", [])

        # Record today's forecast
        new_forecast_record = evaluator.record_forecast(
            target_date_str, "GOLD", "5d", gold_forecasts["research_bias"],
            gold_forecasts["horizons"]["5d"]["expected_median_return_pct"], gold_forecasts["confidence_score"]
        )
        saved_forecasts.append(new_forecast_record)
        self.storage.save_json({"records": saved_forecasts}, "predictions/daily/history.json")

        eval_res = evaluator.evaluate_historical_forecasts(gold_df, saved_forecasts)
        evaluator.update_model_performance_summary(eval_res)

        # 11. Report Generation
        logger.info("11. Generating daily Markdown and JSON reports...")
        json_report_path = ReportGenerator.generate_json_report(
            target_date_str, gold_summary, silver_summary, ratio_summary, macro_summary,
            gold_scores, silver_scores, gold_forecasts, silver_forecasts, gold_analogues, silver_analogues, quality_report
        )
        md_report_path = ReportGenerator.generate_markdown_report(
            target_date_str, gold_summary, silver_summary, ratio_summary, macro_summary,
            gold_scores, silver_scores, gold_forecasts, silver_forecasts, gold_analogues, silver_analogues, quality_report
        )

        logger.info(f"=== COMPLETED PIPELINE FOR {target_date_str} successfully ===")
        return {
            "date": target_date_str,
            "json_report": json_report_path,
            "md_report": md_report_path,
            "gold_price": gold_summary.get("close", 0.0),
            "silver_price": silver_summary.get("close", 0.0)
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precious Metals Daily Research Pipeline Runner")
    parser.add_argument("date", nargs="?", default=datetime.now().strftime("%Y-%m-%d"), help="Execution date YYYY-MM-DD")
    args = parser.parse_args()

    pipeline = DailyMetalsPipeline()
    pipeline.run_pipeline(args.date)
