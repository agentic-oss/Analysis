import os
import sys
import json
import yaml
import logging
import argparse
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd

from src.ingestion.collector import DataCollector, get_economic_calendar_events
from src.validation.validator import DataValidator
from src.storage.manager import StorageManager
from src.indicators.technical import TechnicalIndicators
from src.macro.correlations import CrossMarketCorrelations
from src.metals.gold import GoldAnalyzer
from src.metals.silver import SilverAnalyzer
from src.ratios.relative_value import RelativeValueAnalyzer
from src.regimes.macro_regime import MacroRegimeDetector
from src.regimes.market_regime import MarketRegimeDetector
from src.patterns.analogues import HistoricalAnalogueEngine
from src.forecasting.probabilistic import ProbabilisticForwardEngine
from src.features.builder import FeatureBuilder
from src.models.baseline import TimeSeriesModelEvaluator
from src.evaluation.tracker import ForecastTracker
from src.backtesting.engine import BacktestEngine
from src.scoring.composite import CompositeScoreCalculator
from src.scoring.alerts import AlertEngine
from src.reporting.report_generator import DailyReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("PipelineRunner")


def run_pipeline(target_date: str = None) -> Dict[str, Any]:
    if not target_date:
        target_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    logger.info(f"Starting Precious Metals Pipeline Execution for Date: {target_date}")

    # Load Config
    config_path = "config/config.yaml"
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    # 1. Ingestion
    collector = DataCollector(use_fallback=True)
    start_date = (pd.to_datetime(target_date) - pd.Timedelta(days=1000)).strftime("%Y-%m-%d")
    end_date = (pd.to_datetime(target_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info(f"Collecting market data from {start_date} to {end_date}...")
    raw_data = collector.collect_all(start_date=start_date, end_date=end_date)

    # 2. Validation
    validator = DataValidator()
    validated_data, quality_summary = validator.validate_batch(raw_data)

    # 3. Storage
    storage = StorageManager()
    storage.save_raw_batch(validated_data, target_date)
    storage.save_quality_report(quality_summary, target_date)

    for sym, df in validated_data.items():
        cat = "metals" if "GOLD" in sym or "SILVER" in sym or "MCX" in sym else "macro"
        storage.save_processed_dataset(sym, df, category=cat)

    # 4. Technical Indicators
    processed_metals = {}
    processed_macro = {}

    for sym, df in validated_data.items():
        if df.empty:
            continue
        ti_df = TechnicalIndicators.calculate_all(df)
        if "GOLD" in sym or "SILVER" in sym or "MCX" in sym:
            processed_metals[sym] = ti_df
        else:
            processed_macro[sym] = ti_df

    gold_ti = processed_metals.get("GOLD", pd.DataFrame())
    silver_ti = processed_metals.get("SILVER", pd.DataFrame())

    if gold_ti.empty or silver_ti.empty:
        logger.error("Crucial gold or silver market data is missing. Halting pipeline.")
        return {"status": "failed", "reason": "missing_gold_or_silver"}

    # 5. Correlations & Detailed Analysis
    gold_analysis = GoldAnalyzer.analyze(gold_ti, processed_macro)
    silver_analysis = SilverAnalyzer.analyze(silver_ti, gold_ti, processed_macro)

    # 6. Gold/Silver Ratio Relative Value
    ratio_df = RelativeValueAnalyzer.calculate_gold_silver_ratio(gold_ti, silver_ti)
    ratio_analysis = RelativeValueAnalyzer.analyze_ratio_extremes_and_forward_outcomes(ratio_df)

    # 7. Regimes
    macro_regime = MacroRegimeDetector.detect_regime(processed_macro, target_date)
    gold_market_regime = MarketRegimeDetector.detect_market_regime(gold_ti, "GOLD")
    silver_market_regime = MarketRegimeDetector.detect_market_regime(silver_ti, "SILVER")

    # 8. Historical Analogue Search Engine
    analogues, analogue_stats = HistoricalAnalogueEngine.find_analogues(gold_ti, top_n=15)

    # 9. Probabilistic Forward Engine
    fwd_signals = ProbabilisticForwardEngine.generate_forward_signals(gold_ti, analogue_stats)

    # 10. Feature Store Builder
    events_df = get_economic_calendar_events(start_date, end_date)
    gold_features = FeatureBuilder.build_daily_features(
        "GOLD", gold_ti, processed_macro, ratio_df, events_df,
        macro_regime=macro_regime.get("primary_regime"),
        market_regime=gold_market_regime.get("trend_regime")
    )
    if not gold_features.empty:
        storage.save_features(gold_features, target_date)

    # 11. Model Evaluation & Prediction Logging
    tracker = ForecastTracker()

    # Log current date T forecast predictions
    pred_records = []
    if fwd_signals and "horizons" in fwd_signals:
        for h_key, h_info in fwd_signals["horizons"].items():
            h_days = h_info["horizon_trading_days"]
            pred_records.append({
                "prediction_date": target_date,
                "instrument": "GOLD",
                "horizon": h_days,
                "predicted_direction": "UP" if h_info["expected_return_pct"] >= 0 else "DOWN",
                "predicted_return": h_info["expected_return_pct"],
                "confidence": h_info["probability_positive_pct"],
                "market_regime": gold_market_regime.get("trend_regime")
            })

    tracker.log_predictions(pred_records)

    # Evaluate outcomes for past predictions when date T + horizon has elapsed
    tracker.evaluate_realized_outcomes(processed_metals)
    model_perf_summary = tracker.compute_forecast_performance_summary()

    # Model walk-forward baseline check
    walk_forward_metrics = TimeSeriesModelEvaluator.train_and_evaluate_walk_forward(gold_features, horizon=5)

    # 12. Strategy Backtesting
    backtester = BacktestEngine()
    scores_for_bt = CompositeScoreCalculator.calculate_scores(
        gold_ti, processed_macro, ratio_df, analogue_stats
    )
    gold_ti["composite_score"] = scores_for_bt.get("composite_score", 50.0)
    backtest_res = backtester.run_signal_backtest(gold_ti, signal_col="composite_score")

    # Save backtest result
    bt_path = os.path.join(storage.backtests_dir, f"backtest_{target_date}.json")
    with open(bt_path, "w") as f:
        json.dump(backtest_res, f, indent=2)

    # 13. Scoring & Alert Engine
    scores = CompositeScoreCalculator.calculate_scores(
        gold_ti, processed_macro, ratio_df, analogue_stats
    )
    alert_engine = AlertEngine()
    alerts = alert_engine.detect_alerts(
        target_date,
        {"gold": gold_analysis, "silver": silver_analysis},
        ratio_analysis,
        processed_macro
    )
    storage.save_alerts(alerts, target_date)

    # 14. Report Generation
    md_report = DailyReportGenerator.generate_markdown_report(
        target_date, gold_analysis, silver_analysis, ratio_analysis,
        macro_regime, gold_market_regime, silver_market_regime,
        analogues, analogue_stats, fwd_signals, scores, alerts, quality_summary
    )

    report_md_path = os.path.join("reports/daily", f"{target_date}.md")
    os.makedirs(os.path.dirname(report_md_path), exist_ok=True)
    with open(report_md_path, "w") as f:
        f.write(md_report)

    json_report = DailyReportGenerator.generate_json_report(
        target_date, gold_analysis, silver_analysis, ratio_analysis,
        macro_regime, gold_market_regime, silver_market_regime,
        analogues, analogue_stats, fwd_signals, scores, alerts, quality_summary
    )
    storage.save_daily_analysis(json_report, target_date)
    storage.save_daily_predictions(fwd_signals, target_date)

    logger.info(f"Pipeline Execution Complete for {target_date}!")
    logger.info(f"Gold Price: ${gold_analysis.get('price')} | Trend: {gold_analysis.get('trend')} | Composite Score: {scores.get('composite_score')}/100")
    logger.info(f"Markdown Report generated: {report_md_path}")

    return json_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Daily Precious Metals Analysis Pipeline")
    parser.add_argument("date", nargs="?", default=None, help="Target date YYYY-MM-DD")
    args = parser.parse_args()

    run_pipeline(args.date)
