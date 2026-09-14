"""
Daily Pipeline Runner.
Orchestrates end-to-end data ingestion, validation, technical analysis, macro regime detection,
gold/silver relative value, historical analogue search, probabilistic signals, ML walk-forward,
backtesting, forecast evaluation, alert generation, and daily Markdown/JSON reporting.

Usage:
  PYTHONPATH=. python3 src/pipeline/runner.py [YYYY-MM-DD]
"""
import os
import sys
import json
import yaml
import logging
from datetime import datetime, timedelta
import pandas as pd

from src.ingestion.provider import YFinanceMarketDataProvider, SyntheticMarketDataProvider
from src.validation.validator import DataValidator
from src.storage.manager import StorageManager
from src.indicators.technical import TechnicalAnalysisEngine
from src.metals.precious import GoldAnalyzer, SilverAnalyzer
from src.ratios.relative_value import RelativeValueAnalyzer
from src.regimes.macro_regime import MacroRegimeDetector
from src.regimes.market_regime import MarketRegimeDetector
from src.features.builder import FeatureBuilder
from src.patterns.analogue import HistoricalAnalogueEngine
from src.scoring.composite import FactorScorer, AlertDetector
from src.forecasting.signals import ProbabilisticForecastEngine, ModelTrainer, BacktestingEngine
from src.evaluation.evaluator import ForecastEvaluator
from src.reporting.generator import ReportGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(execution_date: str = None, use_synthetic: bool = False):
    """
    Executes daily end-to-end precious-metals research & forecasting pipeline.
    """
    date_str = execution_date or datetime.now().strftime("%Y-%m-%d")
    logger.info(f"Starting Gold & Silver Research Pipeline for date: {date_str}")

    # Load configuration & instruments
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open("config/instruments.json", "r") as f:
        instruments_cfg = json.load(f)

    storage = StorageManager(config["system"]["data_dir"])

    # 1. Select Market Data Provider
    if use_synthetic:
        provider = SyntheticMarketDataProvider()
    else:
        provider = YFinanceMarketDataProvider(
            max_retries=config["ingestion"]["max_retries"],
            backoff_factor=config["ingestion"]["backoff_factor"],
            timeout=config["ingestion"]["timeout_seconds"]
        )

    end_dt = datetime.strptime(date_str, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=config["ingestion"]["default_lookback_days"])
    start_date_str = start_dt.strftime("%Y-%m-%d")

    raw_data = {}
    validated_data = {}
    quality_reports = []
    validator = DataValidator(max_daily_jump_pct=config["validation"]["max_daily_price_jump_pct"])

    # 2. Data Ingestion & Validation for all enabled instruments
    for inst in instruments_cfg["instruments"]:
        if not inst.get("enabled", True):
            continue

        symbol = inst["symbol"]
        df_inst = provider.fetch_historical_prices(symbol, start_date_str, date_str, instrument_meta=inst)

        # Fallback to synthetic if yfinance returned empty
        if df_inst.empty and not use_synthetic:
            logger.warning(f"Falling back to synthetic data provider for symbol: {symbol}")
            syn_provider = SyntheticMarketDataProvider()
            df_inst = syn_provider.fetch_historical_prices(symbol, start_date_str, date_str, instrument_meta=inst)

        df_val, q_report = validator.validate_dataset(df_inst, symbol)
        raw_data[symbol] = df_inst
        validated_data[symbol] = df_val
        quality_reports.append(q_report)

        # Save raw & processed datasets
        clean_name = symbol.replace("=", "_").replace("^", "").replace("-", "_").replace(".", "_")
        storage.save_dataframe(df_val, f"processed/{inst['asset_class']}/{clean_name}.parquet")

    # Combine data quality report
    combined_quality = {
        "date": date_str,
        "symbol_reports": quality_reports,
        "valid_count": sum(r["valid_count"] for r in quality_reports),
        "suspicious_count": sum(r["suspicious_count"] for r in quality_reports),
        "invalid_count": sum(r["invalid_count"] for r in quality_reports),
        "warnings": [w for r in quality_reports for w in r["warnings"]],
        "errors": [e for r in quality_reports for e in r["errors"]]
    }
    storage.save_json(combined_quality, f"quality/{date_str}.json")

    # 3. Technical Indicators
    gold_df = validated_data.get("GC=F", validated_data.get("GLD", pd.DataFrame()))
    silver_df = validated_data.get("SI=F", validated_data.get("SLV", pd.DataFrame()))

    gold_tech = TechnicalAnalysisEngine.compute_all_indicators(gold_df)
    silver_tech = TechnicalAnalysisEngine.compute_all_indicators(silver_df)

    # 4. Gold & Silver Cross-Asset Correlations
    macro_dfs = {
        "dxy": validated_data.get("DX-Y.NYB", pd.DataFrame()),
        "us10y": validated_data.get("^TNX", pd.DataFrame()),
        "oil": validated_data.get("CL=F", pd.DataFrame()),
        "sp500": validated_data.get("^GSPC", pd.DataFrame()),
        "vix": validated_data.get("^VIX", pd.DataFrame()),
        "usdinr": validated_data.get("USDINR=X", pd.DataFrame())
    }

    gold_analyzer = GoldAnalyzer()
    silver_analyzer = SilverAnalyzer()

    gold_analyzed = gold_analyzer.analyze(gold_tech, macro_dfs)
    silver_analyzed = silver_analyzer.analyze(silver_tech, macro_dfs)

    # Save processed metal analytics
    storage.save_dataframe(gold_analyzed, "processed/metals/gold_analytics.parquet")
    storage.save_dataframe(silver_analyzed, "processed/metals/silver_analytics.parquet")

    # 5. Relative Value & Gold/Silver Ratio Analysis
    relative_value_df = RelativeValueAnalyzer.compute_relative_value(gold_analyzed, silver_analyzed)
    rv_outcomes = RelativeValueAnalyzer.calculate_forward_spread_outcomes(relative_value_df)

    # 6. Macro & Market Regimes
    macro_regime_info = MacroRegimeDetector.detect_regime(
        macro_dfs["dxy"], macro_dfs["us10y"], macro_dfs["oil"], macro_dfs["sp500"], macro_dfs["vix"]
    )
    gold_market_regime = MarketRegimeDetector.classify_market_regime(gold_analyzed)
    silver_market_regime = MarketRegimeDetector.classify_market_regime(silver_analyzed)

    # 7. ML Feature Building
    gold_features = FeatureBuilder.build_daily_features(
        gold_analyzed, macro_dfs, relative_value_df,
        macro_regime_str=macro_regime_info["macro_regime"],
        market_regime_str=gold_market_regime["composite_regime"]
    )
    silver_features = FeatureBuilder.build_daily_features(
        silver_analyzed, macro_dfs, relative_value_df,
        macro_regime_str=macro_regime_info["macro_regime"],
        market_regime_str=silver_market_regime["composite_regime"]
    )

    storage.save_dataframe(gold_features, "features/daily/gold_features.parquet")
    storage.save_dataframe(silver_features, "features/daily/silver_features.parquet")

    # 8. Historical Analogue Engine
    analogue_engine = HistoricalAnalogueEngine()
    gold_analogues = analogue_engine.find_analogues(gold_features)
    silver_analogues = analogue_engine.find_analogues(silver_features)

    # 9. Factor Scoring & Alerts
    gold_scores = FactorScorer.calculate_scores(
        gold_analyzed, macro_regime_info, relative_value_df, gold_analogues, config["scoring"]["weights"]
    )
    silver_scores = FactorScorer.calculate_scores(
        silver_analyzed, macro_regime_info, relative_value_df, silver_analogues, config["scoring"]["weights"]
    )

    gold_alerts = AlertDetector.detect_alerts(date_str, "GOLD", gold_analyzed, relative_value_df)
    silver_alerts = AlertDetector.detect_alerts(date_str, "SILVER", silver_analyzed, relative_value_df)
    all_alerts = gold_alerts + silver_alerts
    storage.save_json({"date": date_str, "alerts": all_alerts}, f"analysis/alerts/{date_str}.json")

    # 10. Probabilistic Research Forecasts & Baseline ML
    gold_signals = ProbabilisticForecastEngine.generate_probabilistic_signals(
        "GOLD", gold_features, gold_analogues, gold_scores
    )
    silver_signals = ProbabilisticForecastEngine.generate_probabilistic_signals(
        "SILVER", silver_features, silver_analogues, silver_scores
    )

    trainer = ModelTrainer()
    gold_model_wf = trainer.train_walk_forward(gold_features)
    silver_model_wf = trainer.train_walk_forward(silver_features)

    # 11. Backtesting Strategy Simulation
    backtester = BacktestingEngine(
        transaction_cost_pct=config["backtesting"]["transaction_cost_pct"],
        slippage_pct=config["backtesting"]["slippage_pct"],
        initial_capital=config["backtesting"]["initial_capital"]
    )
    gold_bt_signal = (gold_features["rsi_14"] < 40).astype(int) if "rsi_14" in gold_features.columns else pd.Series(0, index=gold_features.index)
    gold_backtest_results = backtester.run_signal_backtest(gold_features, gold_bt_signal)
    storage.save_json(gold_backtest_results, f"backtests/gold_rsi_strategy_{date_str}.json")

    # 12. Model Performance Tracking / Evaluation
    evaluator = ForecastEvaluator()
    # Save current prediction record for evaluation tracking
    prediction_record = {
        "date": date_str,
        "gold": {
            "price": float(gold_analyzed["close"].iloc[-1]) if not gold_analyzed.empty else 0.0,
            "composite_score": gold_scores["composite_score"],
            "signals": gold_signals.get("signals", {})
        },
        "silver": {
            "price": float(silver_analyzed["close"].iloc[-1]) if not silver_analyzed.empty else 0.0,
            "composite_score": silver_scores["composite_score"],
            "signals": silver_signals.get("signals", {})
        }
    }
    storage.save_json(prediction_record, f"predictions/daily/{date_str}.json")

    # Evaluate predictions across history
    historical_preds = []
    preds_dir = os.path.join(config["system"]["data_dir"], "predictions/daily")
    if os.path.exists(preds_dir):
        for f in os.listdir(preds_dir):
            if f.endswith(".json"):
                p_data = storage.load_json(f"predictions/daily/{f}")
                if p_data:
                    historical_preds.append(p_data)

    model_eval_summary = evaluator.evaluate_predictions(historical_preds, {"GC=F": gold_analyzed, "SI=F": silver_analyzed})
    storage.save_json(model_eval_summary, "models/model-performance.json")

    # 13. Reports Generation (JSON & Markdown)
    json_report = ReportGenerator.generate_json_report(
        date_str=date_str,
        gold_analysis={
            "price": float(gold_analyzed["close"].iloc[-1]) if not gold_analyzed.empty else 0.0,
            "composite_score": gold_scores["composite_score"],
            "sub_scores": gold_scores["sub_scores"],
            "signals": gold_signals.get("signals", {}),
            "analogues": gold_analogues
        },
        silver_analysis={
            "price": float(silver_analyzed["close"].iloc[-1]) if not silver_analyzed.empty else 0.0,
            "composite_score": silver_scores["composite_score"],
            "sub_scores": silver_scores["sub_scores"],
            "signals": silver_signals.get("signals", {}),
            "analogues": silver_analogues
        },
        relative_value=rv_outcomes,
        macro_regime=macro_regime_info,
        market_regimes={
            "gold": gold_market_regime,
            "silver": silver_market_regime
        },
        quality_report=combined_quality,
        alerts=all_alerts
    )
    storage.save_json(json_report, f"analysis/daily/{date_str}.json")

    md_report = ReportGenerator.generate_markdown_report(
        date_str=date_str,
        gold_analysis=json_report["gold"],
        silver_analysis=json_report["silver"],
        relative_value=rv_outcomes,
        macro_regime=macro_regime_info,
        gold_regime=gold_market_regime,
        silver_regime=silver_market_regime,
        gold_analogues=gold_analogues,
        silver_analogues=silver_analogues,
        quality_report=combined_quality,
        alerts=all_alerts
    )

    reports_dir = os.path.join(config["system"]["reports_dir"], "daily")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, f"{date_str}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_report)

    logger.info(f"Daily Research Pipeline completed successfully. Report saved to: {report_path}")
    return report_path


if __name__ == "__main__":
    exec_date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    run_pipeline(execution_date=exec_date, use_synthetic=False)
