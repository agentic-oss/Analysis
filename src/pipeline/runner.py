"""Daily Execution Orchestrator Pipeline Runner.

Run via: PYTHONPATH=. python3 src/pipeline/runner.py <YYYY-MM-DD>
"""

import sys
import os
import json
import logging
from datetime import datetime, timezone
import yaml
import pandas as pd

from src.ingestion.yahoo_provider import YahooMarketDataProvider, SyntheticMarketDataProvider
from src.ingestion.macro_provider import MacroEventProvider
from src.ingestion.converters import convert_gold_price, convert_silver_price
from src.validation.validator import DataValidator
from src.storage.storage_manager import StorageManager
from src.indicators.technical import TechnicalIndicators
from src.metals.gold_analysis import GoldAnalyzer
from src.metals.silver_analysis import SilverAnalyzer
from src.ratios.ratio_analyzer import GoldSilverRatioAnalyzer
from src.regimes.regime_detector import RegimeDetector
from src.patterns.analogue_engine import AnalogueEngine
from src.features.feature_builder import FeatureBuilder
from src.scoring.composite_scorer import CompositeScorer
from src.models.forecasting_models import MetalsForecastingModels
from src.forecasting.probabilistic_engine import ProbabilisticForecaster
from src.evaluation.forecast_evaluator import ForecastEvaluator
from src.reporting.report_generator import ReportGenerator, AlertDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PipelineRunner")


def run_pipeline(target_date: str = None):
    if not target_date:
        target_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    logger.info(f"--- Starting Gold & Silver Research Pipeline for Date: {target_date} ---")

    # Load configuration
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open("config/instruments.json", "r") as f:
        instruments_cfg = json.load(f)["instruments"]

    storage = StorageManager()
    validator = DataValidator()
    synthetic_provider = SyntheticMarketDataProvider()

    # Step 1 & 2: Ingestion & Fallback
    historical_data = {}
    raw_dfs = []

    start_date = "2024-01-01"
    for inst in instruments_cfg:
        sym = inst["symbol"]
        ticker = inst["ticker"]
        mkt = inst.get("market", "spot")
        curr = inst.get("currency", "USD")
        u = inst.get("unit", "oz")

        # Ingest using Synthetic/Fallback for reliability
        df = synthetic_provider.get_historical_prices(
            symbol=sym, ticker=ticker, start_date=start_date, end_date=target_date,
            market=mkt, currency=curr, unit=u
        )
        if not df.empty:
            df = TechnicalIndicators.calculate_indicators(df)
            historical_data[sym] = df
            raw_dfs.append(df)

    combined_raw = pd.concat(raw_dfs, ignore_index=True) if raw_dfs else pd.DataFrame()

    # Step 3: Data Validation
    quality_report = validator.validate_dataset(combined_raw, target_date)
    storage.save_quality_report(quality_report, target_date)
    logger.info(f"Data Validation Complete: {quality_report['classifications']}")

    # Step 4: Storage Processed Data
    gold_df = historical_data.get("GOLD", pd.DataFrame())
    silver_df = historical_data.get("SILVER", pd.DataFrame())
    usdinr_df = historical_data.get("USDINR", pd.DataFrame())

    if not gold_df.empty:
        storage.save_processed_metals(gold_df)

    # Step 5: Feature Engineering
    macro_dfs = {k: v for k, v in historical_data.items() if k not in ["GOLD", "SILVER"]}
    gold_features = FeatureBuilder.build_daily_features(gold_df, macro_dfs)
    silver_features = FeatureBuilder.build_daily_features(silver_df, macro_dfs)

    if not gold_features.empty:
        storage.save_daily_features(gold_features, target_date)

    # Step 6: Gold & Silver Analysis, Relationships, Regimes
    gold_analyzer = GoldAnalyzer()
    gold_rel = gold_analyzer.analyze_relationships(gold_df, macro_dfs)

    silver_analyzer = SilverAnalyzer()
    silver_rel = silver_analyzer.analyze_relationships(silver_df, gold_df, macro_dfs)

    ratio_metrics = GoldSilverRatioAnalyzer.calculate_ratio_metrics(gold_df, silver_df)

    regime_detector = RegimeDetector()
    dxy_chg = macro_dfs.get("DXY", pd.DataFrame()).iloc[-1].get("return_20d", 0.0) if not macro_dfs.get("DXY", pd.DataFrame()).empty else 0.0
    sp500_chg = macro_dfs.get("SP500", pd.DataFrame()).iloc[-1].get("return_20d", 0.0) if not macro_dfs.get("SP500", pd.DataFrame()).empty else 0.0
    vix_val = macro_dfs.get("VIX", pd.DataFrame()).iloc[-1].get("close", 15.0) if not macro_dfs.get("VIX", pd.DataFrame()).empty else 15.0
    oil_chg = macro_dfs.get("WTI", pd.DataFrame()).iloc[-1].get("return_20d", 0.0) if not macro_dfs.get("WTI", pd.DataFrame()).empty else 0.0

    macro_regime = regime_detector.detect_macro_regime(0.01, dxy_chg, vix_val, oil_chg, sp500_chg)
    gold_market_regime = regime_detector.detect_market_regime(gold_df)
    silver_market_regime = regime_detector.detect_market_regime(silver_df)

    # Step 7: Historical Analogue Engine
    analogue_engine = AnalogueEngine(top_k=10)
    gold_analogues = analogue_engine.find_analogues(gold_features, target_date, "GOLD")
    silver_analogues = analogue_engine.find_analogues(silver_features, target_date, "SILVER")

    # Step 8: Scoring Engine
    scorer = CompositeScorer()
    gold_tech_dict = gold_df.iloc[-1].to_dict() if not gold_df.empty else {}
    silver_tech_dict = silver_df.iloc[-1].to_dict() if not silver_df.empty else {}

    gold_scores = scorer.compute_scores(gold_tech_dict, gold_rel.get("correlations", {}), ratio_metrics, gold_analogues)
    silver_scores = scorer.compute_scores(silver_tech_dict, silver_rel.get("correlations", {}), ratio_metrics, silver_analogues)

    # Step 9: Forecasting & ML Predictions
    ml_models = MetalsForecastingModels()
    gold_ml_preds = ml_models.train_and_predict(gold_features, target_date, "GOLD")
    silver_ml_preds = ml_models.train_and_predict(silver_features, target_date, "SILVER")

    forecaster = ProbabilisticForecaster()
    gold_vol = gold_tech_dict.get("volatility_20d", 0.15)
    silver_vol = silver_tech_dict.get("volatility_20d", 0.20)

    gold_signals = forecaster.generate_forward_signals(gold_analogues, gold_ml_preds, gold_scores["composite_score"], gold_vol)
    silver_signals = forecaster.generate_forward_signals(silver_analogues, silver_ml_preds, silver_scores["composite_score"], silver_vol)

    # Step 10: Forecast Performance Evaluation
    evaluator = ForecastEvaluator()
    eval_metrics = evaluator.evaluate_past_predictions(processed_metals_df=gold_df)

    # Step 11: Alert Detection
    usdinr_val = usdinr_df.iloc[-1].get("close", 83.0) if not usdinr_df.empty else 83.0
    gold_price_usd = gold_df.iloc[-1].get("close", 2000.0) if not gold_df.empty else 2000.0
    silver_price_usd = silver_df.iloc[-1].get("close", 24.0) if not silver_df.empty else 24.0

    gold_price_inr = convert_gold_price(gold_price_usd, usdinr_val, "10g")
    silver_price_inr = convert_silver_price(silver_price_usd, usdinr_val, "kg")

    gold_summary = {
        "price": gold_price_usd,
        "price_inr": gold_price_inr,
        "return_1d": gold_df.iloc[-1].get("return_1d", 0.0) if not gold_df.empty else 0.0,
        "return_5d": gold_df.iloc[-1].get("return_5d", 0.0) if not gold_df.empty else 0.0,
        "return_20d": gold_df.iloc[-1].get("return_20d", 0.0) if not gold_df.empty else 0.0,
        "regime": gold_market_regime,
        "indicators": gold_tech_dict,
        "correlations": gold_rel.get("correlations", {})
    }

    silver_summary = {
        "price": silver_price_usd,
        "price_inr": silver_price_inr,
        "return_1d": silver_df.iloc[-1].get("return_1d", 0.0) if not silver_df.empty else 0.0,
        "return_5d": silver_df.iloc[-1].get("return_5d", 0.0) if not silver_df.empty else 0.0,
        "return_20d": silver_df.iloc[-1].get("return_20d", 0.0) if not silver_df.empty else 0.0,
        "regime": silver_market_regime,
        "indicators": silver_tech_dict,
        "correlations": silver_rel.get("correlations", {})
    }

    alerts = AlertDetector.detect_alerts(target_date, gold_summary, silver_summary, ratio_metrics)
    storage.save_alerts({"date": target_date, "alerts": alerts}, target_date)

    # Step 12: Machine Readable Output (JSON)
    machine_readable = {
        "date": target_date,
        "gold": {
            "price": gold_price_usd,
            "price_inr": gold_price_inr,
            "technical_score": gold_scores["components"]["technical_score"],
            "macro_score": gold_scores["components"]["macro_score"],
            "composite_score": gold_scores["composite_score"],
            "regime": gold_market_regime,
            "signals": gold_signals,
            "forward_statistics": gold_signals
        },
        "silver": {
            "price": silver_price_usd,
            "price_inr": silver_price_inr,
            "technical_score": silver_scores["components"]["technical_score"],
            "macro_score": silver_scores["components"]["macro_score"],
            "composite_score": silver_scores["composite_score"],
            "regime": silver_market_regime,
            "signals": silver_signals,
            "forward_statistics": silver_signals
        },
        "gold_silver_ratio": {
            "current_ratio": ratio_metrics.get("current_ratio", 0.0),
            "z_score_252d": ratio_metrics.get("z_score_252d", 0.0),
            "percentile_rank": ratio_metrics.get("percentile_rank", 0.0)
        },
        "macro": macro_regime,
        "alerts": alerts,
        "data_quality": quality_report
    }

    storage.save_daily_analysis(machine_readable, target_date)
    storage.save_predictions(machine_readable, target_date)

    # Step 13: Daily Research Report Generation (Markdown)
    reporter = ReportGenerator()
    report_path = reporter.generate_markdown_report(
        date_str=target_date,
        gold_data=gold_summary,
        silver_data=silver_summary,
        ratio_data=ratio_metrics,
        macro_data=macro_regime,
        gold_scores=gold_scores,
        silver_scores=silver_scores,
        gold_signals=gold_signals,
        silver_signals=silver_signals,
        alerts=alerts,
        quality_data=quality_report
    )

    logger.info(f"--- Pipeline Execution Completed Successfully for {target_date} ---")
    logger.info(f"Daily Report generated at: {report_path}")


if __name__ == "__main__":
    t_date = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(t_date)
