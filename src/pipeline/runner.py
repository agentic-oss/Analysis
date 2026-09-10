import sys
import os
import argparse
import logging
import json
import pandas as pd
from datetime import datetime, timedelta

from src.ingestion.provider import YFinanceMarketDataProvider, MockMarketDataProvider
from src.ingestion.engine import DataIngestionEngine
from src.validation.validator import DataValidator
from src.storage.storage import DataStorage
from src.metals.analytics import GoldAnalyzer, SilverAnalyzer, RelativeValueAnalyzer
from src.regimes.classifier import MacroRegimeClassifier
from src.patterns.analogue import AnalogueEngine, ForwardProbabilityEngine
from src.features.builder import FeatureStoreBuilder
from src.scoring.scoring import TransparentScoringSystem, AlertEngine
from src.reporting.generator import DailyReportGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("pipeline_runner")

def run_pipeline(target_date: str, use_mock: bool = False):
    logger.info(f"--- Running Precious Metals Pipeline for Date: {target_date} ---")

    # 1. Initialize Components
    storage = DataStorage()
    validator = DataValidator()

    if use_mock:
        provider = MockMarketDataProvider()
    else:
        provider = YFinanceMarketDataProvider()

    engine = DataIngestionEngine(provider=provider)

    # 2. Determine Lookback Range (2 years back to establish indicators & analogues)
    end_dt = datetime.strptime(target_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=730)
    start_date_str = start_dt.strftime("%Y-%m-%d")

    # 3. Ingest Market Data
    logger.info(f"Ingesting market data from {start_date_str} to {target_date}...")
    raw_datasets = engine.fetch_all_instruments(start_date_str, target_date)

    if not raw_datasets or "GOLD" not in raw_datasets or raw_datasets["GOLD"].empty:
        logger.warning("Main gold market data empty or missing, using MockMarketDataProvider fallback for complete dataset generation...")
        fallback_provider = MockMarketDataProvider()
        fallback_engine = DataIngestionEngine(provider=fallback_provider)
        raw_datasets = fallback_engine.fetch_all_instruments(start_date_str, target_date)

    # Convert prices to INR
    gold_raw = raw_datasets.get("GOLD", pd.DataFrame())
    silver_raw = raw_datasets.get("SILVER", pd.DataFrame())
    usdinr_raw = raw_datasets.get("USDINR", pd.DataFrame())

    inr_derived = engine.calculate_inr_converted_prices(gold_raw, silver_raw, usdinr_raw)
    raw_datasets.update(inr_derived)

    # 4. Save Raw & Update Processed Datasets
    for symbol, df in raw_datasets.items():
        if not df.empty:
            storage.save_raw_data(df, target_date, symbol)
            category = "metals" if "GOLD" in symbol or "SILVER" in symbol else "macro"
            storage.save_processed_data(df, category, symbol)

    # 5. Data Validation & Quality Report
    logger.info("Validating dataset quality...")
    quality_report = validator.generate_daily_quality_report(target_date, raw_datasets)

    # 6. Economic Events
    events_df = provider.get_economic_events(start_date_str, target_date)

    # 7. Analytics & Regimes
    logger.info("Running technical and macro analytics...")
    gold_analyzer = GoldAnalyzer()
    silver_analyzer = SilverAnalyzer()
    rv_analyzer = RelativeValueAnalyzer()

    gold_summary = gold_analyzer.analyze(gold_raw, macro_dfs={"DXY": raw_datasets.get("DXY", pd.DataFrame())})
    silver_summary = silver_analyzer.analyze(silver_raw, gold_df=gold_raw)
    rv_summary = rv_analyzer.analyze(gold_raw, silver_raw)

    macro_classifier = MacroRegimeClassifier()
    us10y_df = raw_datasets.get("US10Y", pd.DataFrame())
    dxy_df = raw_datasets.get("DXY", pd.DataFrame())
    oil_df = raw_datasets.get("CRUDE_OIL", pd.DataFrame())
    sp_df = raw_datasets.get("SP500", pd.DataFrame())
    vix_df = raw_datasets.get("VIX", pd.DataFrame())

    us10y_chg = (us10y_df["close"].iloc[-1] - us10y_df["close"].iloc[-20]) if len(us10y_df) >= 20 else 0.0
    dxy_ret = dxy_df["close"].pct_change(20).iloc[-1] if len(dxy_df) >= 20 else 0.0
    oil_ret = oil_df["close"].pct_change(20).iloc[-1] if len(oil_df) >= 20 else 0.0
    sp_ret = sp_df["close"].pct_change(20).iloc[-1] if len(sp_df) >= 20 else 0.0
    vix_lvl = vix_df["close"].iloc[-1] if not vix_df.empty else 18.0

    macro_summary = macro_classifier.classify_macro_regime(
        us10y_change_20d=float(us10y_chg),
        dxy_return_20d=float(dxy_ret),
        oil_return_20d=float(oil_ret),
        sp500_return_20d=float(sp_ret),
        vix_level=float(vix_lvl)
    )

    # 8. Feature Engineering
    logger.info("Building ML feature store...")
    builder = FeatureStoreBuilder()
    macro_dfs = {
        "USDINR": usdinr_raw,
        "DXY": dxy_df,
        "US10Y": us10y_df,
        "CRUDE_OIL": oil_df,
        "SP500": sp_df,
        "VIX": vix_df
    }
    gold_features = builder.create_features_for_instrument("GOLD", gold_raw, macro_dfs=macro_dfs, events_df=events_df)
    if not gold_features.empty:
        storage.save_features(gold_features, target_date)

    # 9. Analogue Search & Probabilities
    logger.info("Searching historical analogues...")
    analogue_engine = AnalogueEngine(top_k=25)
    prob_engine = ForwardProbabilityEngine()

    analogue_summary = {"sample_count": 0, "analogues": [], "forward_stats": {}}
    forecast_signals = {"disclaimer": "Historical statistics.", "signals": {}}

    if not gold_features.empty and len(gold_features) > 20:
        target_row = gold_features.iloc[-1]
        hist_rows = gold_features.iloc[:-1]
        feature_cols = ["rsi_14", "dist_sma_20", "volatility_20d", "macd"]

        analogue_summary = analogue_engine.find_analogues(
            target_features=target_row,
            historical_features_df=hist_rows,
            feature_cols=feature_cols,
            horizons=[1, 3, 5, 10, 20, 60]
        )
        forecast_signals = prob_engine.generate_probabilistic_signals(analogue_summary)

    # 10. Scoring & Alerts
    logger.info("Computing transparent scores and detecting alerts...")
    scorer = TransparentScoringSystem()
    scores = scorer.compute_scores(
        rsi_14=gold_summary.get("rsi_14", 50.0),
        dist_sma_50=(gold_summary.get("price", 0.0) - gold_summary.get("sma_50", gold_summary.get("price", 0.0))) / max(gold_summary.get("sma_50", 1.0), 1.0),
        macd_hist=0.0,
        macro_regime=macro_summary.get("primary_macro_regime", "Neutral"),
        volatility_20d=gold_summary.get("volatility_20d", 0.15),
        gs_ratio_zscore=rv_summary.get("ratio_zscore_60", 0.0),
        analogue_pos_prob=forecast_signals.get("signals", {}).get("5d", {}).get("positive_return_probability", 0.50)
    )

    alerts = AlertEngine.detect_alerts(target_date, gold_raw, silver_raw, macro_dfs)

    # 11. Reports Generation
    logger.info("Generating daily Markdown and JSON reports...")
    report_gen = DailyReportGenerator()
    md_path = report_gen.generate_markdown_report(
        target_date, gold_summary, silver_summary, macro_summary, rv_summary, analogue_summary, forecast_signals, quality_report
    )
    json_path = report_gen.generate_json_report(
        target_date, gold_summary, silver_summary, macro_summary, rv_summary, scores, forecast_signals, quality_report
    )

    logger.info(f"--- Pipeline completed successfully for {target_date} ---")
    return {
        "date": target_date,
        "markdown_report": md_path,
        "json_report": json_path,
        "alerts_count": len(alerts)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precious Metals Daily Research Pipeline Runner")
    parser.add_argument("date", nargs="?", default=datetime.now().strftime("%Y-%m-%d"), help="Target date YYYY-MM-DD")
    parser.add_argument("--mock", action="store_true", help="Use mock provider for deterministic offline execution")
    args = parser.parse_args()

    run_pipeline(args.date, use_mock=args.mock)
