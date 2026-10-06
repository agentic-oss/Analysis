import os
import sys
import datetime
import logging
import yaml
import pandas as pd

from src.ingestion.pipeline import DataIngestionPipeline
from src.validation.validator import DataValidator
from src.storage.manager import DataStorageManager
from src.metals.analyzer import MetalsAnalyzer
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PipelineRunner")

def run_daily_pipeline(target_date_str: str = None):
    if not target_date_str:
        target_date_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")

    logger.info(f"========== Starting Precious Metals Pipeline for {target_date_str} ==========")

    # Load configuration
    config_path = "config/config.yaml"
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    storage = DataStorageManager(
        base_dir=config.get("system", {}).get("data_dir", "data"),
        reports_dir=config.get("system", {}).get("reports_dir", "reports")
    )
    validator = DataValidator()

    # Calculate historical start date (2 years lookback)
    target_dt = pd.to_datetime(target_date_str)
    start_dt = target_dt - pd.Timedelta(days=730)
    start_date_str = start_dt.strftime("%Y-%m-%d")

    # 1. Ingestion
    logger.info("Step 1: Fetching Market & Macro Data...")
    ingestion_pipeline = DataIngestionPipeline()
    raw_datasets = ingestion_pipeline.fetch_all(start_date_str, target_date_str)

    # 2. Validation & Quality Tracking
    logger.info("Step 2: Validating Data Batches...")
    validated_datasets = {}
    validation_summaries = {}

    for key, df in raw_datasets.items():
        if df is not None and not df.empty and "open" in df.columns:
            v_df, summary = validator.validate_dataset(df, key)
            validated_datasets[key] = v_df
            validation_summaries[key] = summary
            # Save raw observation
            storage.save_raw_data(target_date_str, key, v_df)
        else:
            validated_datasets[key] = df

    quality_report = validator.generate_quality_report(target_date_str, validation_summaries)
    storage.save_quality_report(target_date_str, quality_report)

    # 3. Technical & Metals Analysis
    logger.info("Step 3: Calculating Technical Indicators & Metals Analysis...")
    gold_df = MetalsAnalyzer.analyze_metal(validated_datasets.get("gold_usd", pd.DataFrame()), "GOLD")
    silver_df = MetalsAnalyzer.analyze_metal(validated_datasets.get("silver_usd", pd.DataFrame()), "SILVER")

    if not gold_df.empty:
        storage.save_processed_dataset("metals", "gold_usd", gold_df)
    if not silver_df.empty:
        storage.save_processed_dataset("metals", "silver_usd", silver_df)

    gold_summary = MetalsAnalyzer.get_summary_metrics(gold_df)
    silver_summary = MetalsAnalyzer.get_summary_metrics(silver_df)

    # 4. Relative Value & Macro Analysis
    logger.info("Step 4: Analyzing Gold/Silver Ratio & Cross-Market Correlations...")
    ratio_df = GoldSilverRatioAnalyzer.analyze_ratio(gold_df, silver_df)
    ratio_summary = GoldSilverRatioAnalyzer.get_latest_summary(ratio_df)

    macro_dfs = {k: v for k, v in validated_datasets.items() if k not in ["gold_usd", "silver_usd"]}
    macro_corrs = MacroCorrelationAnalyzer.compute_correlations(gold_df, macro_dfs)

    # 5. Regimes & Historical Analogue Search
    logger.info("Step 5: Detecting Macro/Market Regimes & Analogue Search...")
    macro_regime = MacroRegimeDetector.classify_regime(macro_dfs, target_date_str)
    gold_mkt_regime = MarketRegimeDetector.classify_market_regime(gold_df)
    silver_mkt_regime = MarketRegimeDetector.classify_market_regime(silver_df)

    # 6. Feature Store & Forecast Models
    logger.info("Step 6: Generating Daily Feature Store & Running Forecasting Models...")
    gold_features = DailyFeatureStoreGenerator.generate_features(gold_df, macro_dfs, "GOLD")
    silver_features = DailyFeatureStoreGenerator.generate_features(silver_df, macro_dfs, "SILVER")

    analogues = HistoricalAnalogueEngine.find_analogues(gold_features, target_date_str, top_k=10)
    fwd_probs = ForwardProbabilityEngine.calculate_forward_probabilities(gold_features, target_date_str, "GOLD")

    models = ForecastingModels()
    gold_model_preds = models.train_and_predict(gold_features, target_date_str, horizon=5)

    # Track predictions for performance measurement
    tracker = ForecastTracker()
    tracker.record_prediction(
        prediction_date=target_date_str,
        instrument="GOLD",
        horizon=5,
        predicted_dir=gold_model_preds.get("predicted_direction", "Bullish"),
        predicted_ret=gold_model_preds.get("predicted_return", 0.0),
        confidence=75.0
    )
    if not gold_df.empty:
        tracker.update_actual_outcomes(gold_df, "GOLD")

    # 7. Backtesting Strategy Verification
    logger.info("Step 7: Executing Signal Backtest...")
    if not gold_features.empty:
        gold_features["signal"] = gold_features["rsi_14"].apply(lambda r: 1 if r < 40 else (-1 if r > 60 else 0))
        bt_engine = BacktestingEngine()
        bt_results = bt_engine.run_backtest(gold_features, "signal")
    else:
        bt_results = {}

    # 8. Composite Scoring & Alerts
    logger.info("Step 8: Computing Composite Scores & Triggering Alerts...")
    scorer = TransparentScoringSystem()
    scores = scorer.compute_scores(
        tech_metrics=gold_summary,
        macro_info=macro_regime,
        ratio_info=ratio_summary,
        analogue_info=analogues,
        model_info=gold_model_preds
    )

    alert_detector = AlertDetector()
    alerts = alert_detector.detect_alerts(target_date_str, gold_summary, ratio_summary, "GOLD")
    storage.save_alerts_json(target_date_str, alerts)

    # 9. Daily Reporting
    logger.info("Step 9: Generating Machine-Readable JSON & Markdown Reports...")
    json_report = ReportGenerator.generate_json_report(
        date_str=target_date_str,
        gold_summary={"metrics": gold_summary, "regime": gold_mkt_regime, "scores": scores, "predictions": gold_model_preds},
        silver_summary={"metrics": silver_summary, "regime": silver_mkt_regime},
        ratio_summary=ratio_summary,
        macro_summary=macro_regime,
        quality_summary=quality_report
    )
    storage.save_analysis_json(target_date_str, json_report)

    gold_summary["research_bias"] = gold_model_preds.get("predicted_direction", "Neutral")
    silver_summary["research_bias"] = "Neutral"

    md_report = ReportGenerator.generate_markdown_report(
        date_str=target_date_str,
        gold_info=gold_summary,
        silver_info=silver_summary,
        ratio_info=ratio_summary,
        macro_info=macro_regime,
        analogues_info=analogues,
        fwd_prob_info=fwd_probs,
        quality_info=quality_report
    )
    storage.save_daily_report_md(target_date_str, md_report)

    logger.info(f"========== Daily Pipeline Completed Successfully for {target_date_str} ==========")
    return json_report

if __name__ == "__main__":
    date_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_daily_pipeline(date_arg)
