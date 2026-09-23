import os
import sys
import json
import yaml
import datetime
import logging
import pandas as pd
import numpy as np

from src.ingestion.provider import YFinanceProvider, SyntheticProvider, convert_gold_usd_oz_to_inr_10g, convert_silver_usd_oz_to_inr_kg
from src.validation.validator import DataValidator, save_quality_report
from src.storage.manager import StorageManager
from src.indicators.technical import calculate_technical_indicators
from src.macro.analyzer import MetalsMacroAnalyzer
from src.ratios.relative_value import RelativeValueAnalyzer
from src.regimes.classifier import RegimeClassifier
from src.patterns.analogue import HistoricalAnalogueEngine
from src.features.builder import FeatureStoreBuilder
from src.scoring.composite import TransparentScorer
from src.forecasting.engine import ForecasterPipeline
from src.backtesting.engine import StrategyBacktester
from src.evaluation.tracker import ForecastEvaluator
from src.reporting.generator import ReportGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PipelineRunner")

def run_pipeline(target_date: str, use_synthetic: bool = False):
    logger.info(f"Starting Precious Metals Pipeline for Target Date: {target_date}")

    # Load Configurations
    with open("config/instruments.json", "r") as f:
        instruments_cfg = json.load(f)["instruments"]
    with open("config/config.yaml", "r") as f:
        config_cfg = yaml.safe_load(f)

    storage = StorageManager()
    validator = DataValidator()

    # Provider Initialization
    if use_synthetic:
        provider = SyntheticProvider()
    else:
        provider = YFinanceProvider(
            retries=config_cfg["ingestion"]["retries"],
            backoff_factor=config_cfg["ingestion"]["backoff_factor"],
            timeout=config_cfg["ingestion"]["timeout"]
        )

    dt_target = pd.to_datetime(target_date)
    dt_start = (dt_target - pd.DateOffset(years=config_cfg["ingestion"]["history_years"])).strftime("%Y-%m-%d")

    collected_data = {}
    quality_summary = {"valid_records": 0, "suspicious_records": 0, "invalid_records": 0}

    # 1. Ingestion & Validation
    for inst in instruments_cfg:
        if not inst.get("enabled", True):
            continue
        symbol = inst["symbol"]
        ticker = inst.get("yfinance_ticker")
        category = inst.get("category", "metals")

        try:
            if ticker or use_synthetic:
                raw_df = provider.get_historical_prices(symbol, dt_start, target_date, ticker=ticker)
            else:
                raw_df = pd.DataFrame()

            if not raw_df.empty:
                val_df, q_report = validator.validate_dataset(raw_df, symbol, target_date)
                storage.save_raw_observation(val_df, symbol, target_date)
                storage.save_processed_data(val_df, category, symbol)
                save_quality_report(q_report, target_date)

                quality_summary["valid_records"] += q_report["valid_records"]
                quality_summary["suspicious_records"] += q_report["suspicious_records"]
                quality_summary["invalid_records"] += q_report["invalid_records"]

                collected_data[symbol] = val_df
        except Exception as e:
            logger.warning(f"Failed ingestion for {symbol}: {e}")

    # Fallback to synthetic if required primary assets missing
    if "GOLD" not in collected_data or collected_data["GOLD"].empty:
        logger.info("GOLD data missing or yfinance offline, using synthetic provider fallback.")
        synth = SyntheticProvider()
        for inst in instruments_cfg:
            sym = inst["symbol"]
            cat = inst.get("category", "metals")
            df = synth.get_historical_prices(sym, dt_start, target_date)
            val_df, q_report = validator.validate_dataset(df, sym, target_date)
            storage.save_raw_observation(val_df, sym, target_date)
            storage.save_processed_data(val_df, cat, sym)
            collected_data[sym] = val_df

    # Indian Currency Conversions
    gold_df = collected_data.get("GOLD", pd.DataFrame())
    silver_df = collected_data.get("SILVER", pd.DataFrame())
    usdinr_df = collected_data.get("USDINR", pd.DataFrame())

    if not gold_df.empty and not usdinr_df.empty:
        merged_gold = pd.merge(gold_df[["timestamp", "close"]], usdinr_df[["timestamp", "close"]], on="timestamp", suffixes=("_gold", "_usdinr"))
        merged_gold["close_inr_10g"] = merged_gold.apply(lambda r: convert_gold_usd_oz_to_inr_10g(r["close_gold"], r["close_usdinr"]), axis=1)
        storage.save_processed_data(merged_gold, "metals", "GOLD_INR_CONVERTED")

    # 2. Indicators & Relationships
    gold_ind = calculate_technical_indicators(gold_df)
    silver_ind = calculate_technical_indicators(silver_df)

    macro_dfs = {k: v for k, v in collected_data.items() if k not in ["GOLD", "SILVER"]}
    macro_analyzer = MetalsMacroAnalyzer()
    macro_summary = macro_analyzer.get_latest_correlation_summary(gold_df, macro_dfs)

    rv_analyzer = RelativeValueAnalyzer()
    ratio_df = rv_analyzer.calculate_ratio_series(gold_df, silver_df)
    ratio_summary = rv_analyzer.analyze_ratio_extremes(ratio_df)

    # 3. Regimes & Historical Analogues
    classifier = RegimeClassifier()
    latest_gold = gold_ind.iloc[-1] if not gold_ind.empty else pd.Series()
    latest_silver = silver_ind.iloc[-1] if not silver_ind.empty else pd.Series()

    gold_mkt_regime = classifier.classify_market_regime(latest_gold)
    silver_mkt_regime = classifier.classify_market_regime(latest_silver)

    macro_regime = classifier.classify_macro_regime(
        vix_level=float(collected_data.get("VIX", pd.DataFrame()).iloc[-1]["close"]) if "VIX" in collected_data and not collected_data["VIX"].empty else 15.0
    )

    analogue_engine = HistoricalAnalogueEngine(top_k=10)
    gold_analogues = analogue_engine.find_analogues(gold_ind.iloc[:-10] if len(gold_ind) > 10 else gold_ind, latest_gold)
    silver_analogues = analogue_engine.find_analogues(silver_ind.iloc[:-10] if len(silver_ind) > 10 else silver_ind, latest_silver)

    # 4. Feature Engineering & ML Forecasting
    builder = FeatureStoreBuilder()
    gold_features = builder.build_feature_dataset(gold_df, ratio_df=ratio_df)
    silver_features = builder.build_feature_dataset(silver_df, ratio_df=ratio_df)

    storage.save_processed_data(gold_features, "features", "GOLD_DAILY_FEATURES")
    storage.save_processed_data(silver_features, "features", "SILVER_DAILY_FEATURES")

    forecaster = ForecasterPipeline()
    feat_cols = ["rsi_14", "dist_sma_20", "dist_sma_50", "volatility_20d"]
    gold_forecasts = forecaster.generate_all_forecasts(gold_features, feat_cols)
    silver_forecasts = forecaster.generate_all_forecasts(silver_features, feat_cols)

    # Track predictions for performance monitoring
    historical_predictions = []
    for h, f_data in gold_forecasts.items():
        historical_predictions.append({
            "prediction_date": target_date,
            "instrument": "GOLD",
            "horizon": h,
            "predicted_direction": 1 if f_data["probability_positive"] >= 50.0 else 0,
            "predicted_return": f_data["expected_return"],
            "actual_return": float(latest_gold.get("return_1d", 0.0)) if h == "1d" else None
        })

    evaluator = ForecastEvaluator()
    evaluator.evaluate_predictions(historical_predictions)

    # 5. Composite Scoring
    scorer = TransparentScorer()
    gold_score = scorer.compute_composite_score(latest_gold, macro_summary, ratio_summary, gold_analogues)
    silver_score = scorer.compute_composite_score(latest_silver, macro_summary, ratio_summary, silver_analogues)

    # 6. Backtesting
    backtester = StrategyBacktester()
    signals = (gold_ind["rsi_14"] < 40).astype(int) if "rsi_14" in gold_ind.columns else pd.Series(0, index=gold_df.index)
    backtest_res = backtester.run_backtest(gold_df, signals)
    storage.save_json(backtest_res, "backtests", f"gold_rsi_backtest_{target_date}")

    # 7. Reporting & Machine-Readable Output
    analysis_payload = {
        "date": target_date,
        "gold": {
            "price": float(latest_gold.get("close", 0.0)),
            "daily_return_pct": float(latest_gold.get("return_1d", 0.0) or 0.0),
            "weekly_return_pct": float(latest_gold.get("return_5d", 0.0) or 0.0),
            "monthly_return_pct": float(latest_gold.get("return_20d", 0.0) or 0.0),
            "composite_score": gold_score["composite_score"],
            "component_scores": gold_score["component_scores"],
            "forward_statistics": gold_forecasts
        },
        "silver": {
            "price": float(latest_silver.get("close", 0.0)),
            "daily_return_pct": float(latest_silver.get("return_1d", 0.0) or 0.0),
            "weekly_return_pct": float(latest_silver.get("return_5d", 0.0) or 0.0),
            "monthly_return_pct": float(latest_silver.get("return_20d", 0.0) or 0.0),
            "composite_score": silver_score["composite_score"],
            "component_scores": silver_score["component_scores"],
            "forward_statistics": silver_forecasts
        },
        "gold_silver_ratio": ratio_summary,
        "macro": macro_regime,
        "market_regime": {
            "gold": gold_mkt_regime,
            "silver": silver_mkt_regime
        },
        "data_quality": quality_summary
    }

    reporter = ReportGenerator()
    reporter.generate_daily_markdown_report(target_date, analysis_payload)
    reporter.generate_daily_json_output(target_date, analysis_payload)
    reporter.detect_and_save_alerts(target_date, analysis_payload, config_cfg["alert_thresholds"])

    logger.info("Daily pipeline completed successfully.")

if __name__ == "__main__":
    t_date = sys.argv[1] if len(sys.argv) > 1 else datetime.datetime.now().strftime("%Y-%m-%d")
    use_synth = "--synthetic" in sys.argv
    run_pipeline(t_date, use_synthetic=use_synth)
