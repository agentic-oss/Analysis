import os
import sys
import logging
import json
import pandas as pd
from typing import Dict, Any

from src.ingestion import MarketDataProvider
from src.validation import DataValidator
from src.storage import StorageManager
from src.indicators import calculate_technical_indicators
from src.metals import analyze_gold_drivers
from src.ratios import analyze_gold_silver_ratio
from src.regimes import detect_macro_regime, detect_market_regime
from src.patterns import find_historical_analogues
from src.forecasting import generate_probabilistic_signals
from src.features import build_daily_feature_dataset
from src.scoring import calculate_transparent_scores
from src.evaluation import ForecastEvaluator
from src.reporting import ReportGenerator, detect_daily_alerts

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PipelineRunner")


def run_pipeline(execution_date: str = None) -> Dict[str, Any]:
    if execution_date is None:
        execution_date = pd.Timestamp.now().strftime("%Y-%m-%d")

    logger.info(f"Starting Gold & Silver Pipeline run for date: {execution_date}")

    provider = MarketDataProvider()
    validator = DataValidator()
    storage = StorageManager()
    reporter = ReportGenerator()
    evaluator = ForecastEvaluator()

    # Define date window for historical lookback
    start_date = (pd.to_datetime(execution_date) - pd.Timedelta(days=730)).strftime("%Y-%m-%d")
    end_date = (pd.to_datetime(execution_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # Instruments to fetch
    instruments_to_fetch = [
        ("GOLD_USD_SPOT", "precious_metals", "GOLD"),
        ("SILVER_USD_SPOT", "precious_metals", "SILVER"),
        ("USD_INR", "currency", "USDINR"),
        ("DXY", "currency", "DXY"),
        ("US10Y", "rates", "US10Y"),
        ("US_REAL_YIELD", "rates", "US_REAL_YIELD"),
        ("BRENT_CRUDE", "energy", "BRENT"),
        ("SP500", "equity", "SPX"),
        ("VIX", "volatility", "VIX")
    ]

    fetched_dfs = {}
    quality_summaries = {}

    for inst_id, cat, sym in instruments_to_fetch:
        try:
            df = provider.fetch_historical_ohlcv(inst_id, start_date, end_date)
            v_df, summary = validator.validate_df(df, inst_id)
            storage.save_raw_data(v_df, inst_id, execution_date)
            storage.save_processed_data(v_df, cat, sym)
            fetched_dfs[inst_id] = v_df
            quality_summaries[inst_id] = summary
        except Exception as e:
            logger.error(f"Failed ingestion/validation for {inst_id}: {e}")

    validator.save_quality_report(execution_date, quality_summaries)

    # Load gold and silver processed data
    gold_df = storage.load_processed_data("precious_metals", "GOLD")
    silver_df = storage.load_processed_data("precious_metals", "SILVER")

    if gold_df is None or gold_df.empty or silver_df is None or silver_df.empty:
        logger.error("Insufficient gold or silver data to run analysis.")
        return {"error": "Missing essential precious metals data"}

    # Calculate Technical Indicators
    gold_tech = calculate_technical_indicators(gold_df)
    silver_tech = calculate_technical_indicators(silver_df)

    # Macro & Drivers Analysis
    macro_dfs = {
        "DXY": fetched_dfs.get("DXY"),
        "US10Y": fetched_dfs.get("US10Y"),
        "US_REAL_YIELD": fetched_dfs.get("US_REAL_YIELD"),
        "BRENT": fetched_dfs.get("BRENT_CRUDE"),
        "SP500": fetched_dfs.get("SP500"),
        "VIX": fetched_dfs.get("VIX")
    }

    usd_inr_rate = float(fetched_dfs["USD_INR"]["close"].iloc[-1]) if "USD_INR" in fetched_dfs and not fetched_dfs["USD_INR"].empty else 83.0

    gold_drivers = analyze_gold_drivers(gold_tech, macro_dfs)
    ratio_summary = analyze_gold_silver_ratio(gold_tech, silver_tech, usd_inr=usd_inr_rate)

    # Regimes
    macro_reg = detect_macro_regime(macro_dfs)
    gold_mkt_reg = detect_market_regime(gold_tech)
    silver_mkt_reg = detect_market_regime(silver_tech)

    # Analogues & Probabilistic Forecasting
    gold_analogues = find_historical_analogues(gold_tech)
    silver_analogues = find_historical_analogues(silver_tech)

    gold_signals = generate_probabilistic_signals(gold_analogues, "Gold")
    silver_signals = generate_probabilistic_signals(silver_analogues, "Silver")

    # Feature engineering
    events_df = provider.get_event_calendar(start_date, end_date)
    gold_features = build_daily_feature_dataset(gold_tech, macro_dfs, gold_df=gold_tech, silver_df=silver_tech, events_df=events_df)
    storage.save_features(gold_features, execution_date)

    # Transparent Scoring
    scores = calculate_transparent_scores(
        gold_tech,
        macro_regime=macro_reg["macro_regime"],
        ratio_stats=ratio_summary,
        analogue_stats=gold_analogues
    )

    # Summaries for Reporting
    latest_gold = gold_tech.iloc[-1].to_dict()
    latest_gold["market_regime"] = gold_mkt_reg["combined_regime"]

    latest_silver = silver_tech.iloc[-1].to_dict()
    latest_silver["market_regime"] = silver_mkt_reg["combined_regime"]

    # Save Machine-Readable JSON Analysis
    daily_analysis_json = {
        "date": execution_date,
        "gold": {
            "price": float(latest_gold.get("close", 0.0)),
            "return_1d": float(latest_gold.get("return_1d", 0.0)),
            "market_regime": gold_mkt_reg["combined_regime"],
            "technical_score": scores.get("technical_score", 0.0),
            "signals": gold_signals
        },
        "silver": {
            "price": float(latest_silver.get("close", 0.0)),
            "return_1d": float(latest_silver.get("return_1d", 0.0)),
            "market_regime": silver_mkt_reg["combined_regime"],
            "signals": silver_signals
        },
        "gold_silver_ratio": ratio_summary,
        "macro": macro_reg,
        "scores": scores
    }
    storage.save_daily_analysis(daily_analysis_json, execution_date)

    # Save Predictions JSON
    predictions_json = {
        "date": execution_date,
        "predictions": {
            "GOLD": gold_signals,
            "SILVER": silver_signals
        }
    }
    storage.save_predictions(predictions_json, execution_date)

    # Evaluate Prior Predictions
    evaluator.evaluate_past_predictions(gold_tech, "GOLD")

    # Detect Alerts
    alerts = detect_daily_alerts(execution_date, latest_gold, latest_silver, ratio_summary, macro_reg)
    storage.save_daily_alerts(alerts, execution_date)

    # Generate Markdown Report
    report_file = reporter.generate_daily_report(
        execution_date,
        gold_summary=latest_gold,
        silver_summary=latest_silver,
        ratio_summary=ratio_summary,
        macro_summary=macro_reg,
        gold_signals=gold_signals,
        silver_signals=silver_signals,
        scores=scores,
        data_quality=quality_summaries.get("GOLD_USD_SPOT", {})
    )

    logger.info(f"Pipeline successfully completed! Report generated at: {report_file}")
    return {"status": "SUCCESS", "report": report_file, "date": execution_date}


if __name__ == "__main__":
    date_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(date_arg)
