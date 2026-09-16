import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
import pandas as pd

from src.ingestion.fetcher import DataIngestionEngine
from src.validation.validator import DataValidator
from src.storage.manager import StorageManager
from src.indicators.technical import TechnicalIndicators
from src.metals.analytics import MetalCrossMarketAnalytics
from src.ratios.relative_value import RelativeValueAnalytics
from src.regimes.detector import RegimeDetector
from src.patterns.analogue import HistoricalAnalogueEngine
from src.forecasting.probability import ForwardProbabilityEngine
from src.features.builder import FeatureStoreBuilder
from src.models.baseline import MetalsForecastingModels
from src.scoring.engine import ScoringEngine
from src.backtesting.engine import BacktestEngine
from src.evaluation.tracker import ForecastEvaluator
from src.reporting.generator import DailyReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("DailyMetalsPipeline")


def run_pipeline(execution_date: str) -> None:
    logger.info(f"--- Starting Precious Metals Research Pipeline for {execution_date} ---")

    # 1. Ingestion
    ingestion = DataIngestionEngine()
    datasets = ingestion.fetch_all()
    if not datasets:
        logger.error("No market data fetched. Exiting pipeline.")
        sys.exit(1)

    # 2. Validation & Storage
    validator = DataValidator()
    storage = StorageManager()
    quality_metrics = []

    validated_datasets = {}
    for symbol, df in datasets.items():
        val_df, metrics = validator.validate_dataset(symbol, df)
        quality_metrics.append(metrics)
        category = "macro" if symbol in ["DXY", "US10Y", "US2Y", "CRUDE_OIL", "SP500", "NIFTY50", "VIX", "INDIA_VIX", "USDINR"] else "metals"
        storage.save_processed(val_df, symbol, category=category)
        storage.save_raw_snapshot(val_df, symbol, execution_date)
        validated_datasets[symbol] = val_df

    quality_report = validator.generate_daily_quality_report(quality_metrics, execution_date)

    # 3. Macro Regimes & Cross-Market Analytics
    macro_regime_info = RegimeDetector.detect_macro_regime(validated_datasets)
    gold_df = validated_datasets.get("GOLD", pd.DataFrame())
    silver_df = validated_datasets.get("SILVER", pd.DataFrame())

    if gold_df.empty or silver_df.empty:
        logger.error("Missing primary Gold or Silver dataset. Exiting pipeline.")
        sys.exit(1)

    # Technical Indicators
    gold_tech = TechnicalIndicators.calculate_all(gold_df)
    silver_tech = TechnicalIndicators.calculate_all(silver_df)

    # Correlations
    gold_corr = MetalCrossMarketAnalytics.calculate_rolling_correlations(gold_tech, validated_datasets)
    silver_corr = MetalCrossMarketAnalytics.calculate_rolling_correlations(silver_tech, validated_datasets)

    # Gold/Silver Ratio Analytics
    ratio_df = RelativeValueAnalytics.calculate_gold_silver_ratio(gold_df, silver_df)

    # Market Regimes
    gold_market_regime = RegimeDetector.detect_market_regime(gold_tech)
    silver_market_regime = RegimeDetector.detect_market_regime(silver_tech)

    # 4. Feature Store Builder
    events_df = ingestion.provider.get_economic_indicators()
    feature_builder = FeatureStoreBuilder()
    gold_features = feature_builder.create_features_for_instrument(gold_corr, validated_datasets, events_df=events_df)
    silver_features = feature_builder.create_features_for_instrument(silver_corr, validated_datasets, events_df=events_df)

    storage.save_features(gold_features, execution_date)

    # 5. Historical Analogue Engine & Forward Probabilities
    analogue_engine = HistoricalAnalogueEngine(top_k=20)
    current_gold_state = {
        "rsi_14": float(gold_tech.iloc[-1].get("rsi_14", 50.0)),
        "dist_sma_200": float(gold_tech.iloc[-1].get("dist_sma_200", 0.0)),
        "volatility_20d": float(gold_tech.iloc[-1].get("volatility_20d", 0.15)),
    }
    gold_analogues = analogue_engine.find_analogues(
        gold_features, current_gold_state, feature_cols=["rsi_14", "dist_sma_200", "volatility_20d"]
    )
    gold_fwd_probs = ForwardProbabilityEngine.generate_probabilities(gold_analogues)

    # 6. Forecasting ML Baseline
    model_engine = MetalsForecastingModels(target_horizon=5)
    feature_cols = ["rsi_14", "dist_sma_200", "volatility_20d", "dist_sma_50", "atr_14"]
    model_pred = model_engine.predict_current(gold_features, feature_cols=feature_cols)

    # 7. Scoring Engine
    scoring_engine = ScoringEngine()
    latest_ratio = ratio_df.iloc[-1].to_dict() if not ratio_df.empty else {}
    latest_gold_tech = gold_tech.iloc[-1].to_dict()

    gold_scores = scoring_engine.compute_all_scores(
        technical_metrics=latest_gold_tech,
        macro_metrics=macro_regime_info,
        relative_value_metrics=latest_ratio,
        analogue_metrics=gold_analogues,
        model_metrics=model_pred,
    )

    # 8. Backtest Baseline Strategy
    bt_engine = BacktestEngine()
    gold_tech["signal"] = (gold_tech["rsi_14"] < 30).astype(int) - (gold_tech["rsi_14"] > 70).astype(int)
    bt_results = bt_engine.run_backtest(gold_tech, signal_col="signal", price_col="close")

    # 9. Forecast Performance Tracker
    evaluator = ForecastEvaluator()
    evaluator.log_daily_prediction(
        date_str=execution_date,
        instrument="GOLD",
        horizon=5,
        predicted_direction=model_pred["direction_pred"],
        predicted_return=model_pred["return_pred"],
        confidence=float(gold_scores["confidence_score"]),
    )
    eval_summary = evaluator.evaluate_past_predictions(validated_datasets)

    # 10. Risk Alerts Detection
    alerts = []
    if latest_gold_tech.get("rsi_14", 50) > 70:
        alerts.append({"type": "GOLD_RSI_OVERBOUGHT", "message": f"Gold RSI is overbought at {latest_gold_tech['rsi_14']:.1f}"})
    elif latest_gold_tech.get("rsi_14", 50) < 30:
        alerts.append({"type": "GOLD_RSI_OVERSOLD", "message": f"Gold RSI is oversold at {latest_gold_tech['rsi_14']:.1f}"})

    if latest_ratio.get("gold_silver_ratio", 70) > 85:
        alerts.append({"type": "RATIO_EXTREME_HIGH", "message": f"Gold/Silver ratio extreme high: {latest_ratio['gold_silver_ratio']:.2f}"})

    # 11. Report Generation & JSON Machine-Readable Outputs
    reporter = DailyReportGenerator()
    reporter.generate_alerts(execution_date, alerts)

    payload = {
        "date": execution_date,
        "gold": {
            "price": float(latest_gold_tech.get("close", 0.0)),
            "daily_move_pct": float(latest_gold_tech.get("return_1d", 0.0) * 100.0),
            "weekly_move_pct": float(latest_gold_tech.get("return_5d", 0.0) * 100.0),
            "monthly_move_pct": float(latest_gold_tech.get("return_20d", 0.0) * 100.0),
            "regime": gold_market_regime,
            "bias": gold_fwd_probs.get("5d", {}).get("research_bias", "Neutral"),
            "sma_20": float(latest_gold_tech.get("sma_20", 0.0)),
            "sma_50": float(latest_gold_tech.get("sma_50", 0.0)),
            "sma_200": float(latest_gold_tech.get("sma_200", 0.0)),
            "rsi_14": float(latest_gold_tech.get("rsi_14", 50.0)),
            "atr_14": float(latest_gold_tech.get("atr_14", 0.0)),
            "support_20d": float(gold_tech["low"].rolling(20, min_periods=1).min().iloc[-1]),
            "resistance_20d": float(gold_tech["high"].rolling(20, min_periods=1).max().iloc[-1]),
            "scores": gold_scores,
            "forward_statistics": gold_fwd_probs,
        },
        "silver": {
            "price": float(silver_tech.iloc[-1].get("close", 0.0)),
            "daily_move_pct": float(silver_tech.iloc[-1].get("return_1d", 0.0) * 100.0),
            "weekly_move_pct": float(silver_tech.iloc[-1].get("return_5d", 0.0) * 100.0),
            "monthly_move_pct": float(silver_tech.iloc[-1].get("return_20d", 0.0) * 100.0),
            "regime": silver_market_regime,
            "scores": {"composite_score": 50.0},
            "bias": "Neutral",
        },
        "gold_silver_ratio": {
            "current_ratio": float(latest_ratio.get("gold_silver_ratio", 0.0)),
            "sma_200": float(latest_ratio.get("gs_ratio_sma_200", 0.0)),
            "zscore": float(latest_ratio.get("gs_ratio_zscore", 0.0)),
            "percentile": float(latest_ratio.get("gs_ratio_percentile_252", 50.0)),
        },
        "macro": macro_regime_info,
        "backtest_summary": bt_results,
        "forecast_evaluation": eval_summary,
        "data_quality": quality_report,
        "alerts": alerts,
    }

    json_file = reporter.generate_json_output(execution_date, payload)
    md_file = reporter.generate_markdown_report(execution_date, payload)

    logger.info(f"Pipeline finished successfully. Report generated at {md_file} and JSON at {json_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precious Metals Daily Research Pipeline")
    parser.add_argument(
        "date",
        nargs="?",
        default=datetime.utcnow().strftime("%Y-%m-%d"),
        help="Execution date (YYYY-MM-DD)",
    )
    args = parser.parse_args()
    run_pipeline(args.date)
