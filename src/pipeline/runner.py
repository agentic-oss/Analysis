import os
import sys
import json
import logging
from datetime import datetime, timedelta
import yaml
import pandas as pd

from src.ingestion.yfinance_provider import YFinanceProvider
from src.ingestion.synthetic_provider import SyntheticProvider
from src.validation.validator import DataValidator
from src.storage.storage_manager import StorageManager
from src.indicators.technical import TechnicalAnalysis
from src.macro.macro_analyzer import MacroAnalyzer
from src.metals.gold_analyzer import GoldAnalyzer
from src.metals.silver_analyzer import SilverAnalyzer
from src.ratios.ratio_analyzer import RatioAnalyzer
from src.regimes.market_regime import MarketRegime
from src.patterns.analogue_engine import AnalogueEngine
from src.features.feature_builder import FeatureBuilder
from src.scoring.composite_scorer import CompositeScorer
from src.models.forecaster import MetalsForecaster
from src.forecasting.probabilistic_engine import ProbabilisticEngine
from src.backtesting.backtest_engine import BacktestEngine
from src.evaluation.performance_tracker import ForecastPerformanceTracker
from src.reporting.alert_generator import AlertGenerator
from src.reporting.report_generator import ReportGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PipelineRunner")

def run_pipeline(execution_date: str = None, use_synthetic: bool = False):
    if execution_date is None:
        execution_date = datetime.now().strftime("%Y-%m-%d")

    logger.info(f"--- Starting Precious Metals Pipeline for Date: {execution_date} ---")

    # Load configuration
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open("config/instruments.json", "r") as f:
        instruments_cfg = json.load(f)["instruments"]

    storage = StorageManager(base_dir=config["system"]["data_dir"])
    validator = DataValidator()

    # Provider selection
    if use_synthetic:
        logger.info("Using Synthetic Provider")
        provider = SyntheticProvider()
    else:
        logger.info("Using YFinance Provider")
        provider = YFinanceProvider()

    # Determine date range (lookback)
    end_dt = datetime.strptime(execution_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=365 * config["system"]["historical_lookback_years"])
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = execution_date

    # 1. Collect and Validate Raw Data for all configured instruments
    raw_dfs = {}
    quality_summaries = []

    for inst in instruments_cfg:
        symbol = inst["symbol"]
        ticker = inst["ticker"]
        if not inst.get("enabled", True):
            continue

        logger.info(f"Fetching data for {symbol} ({ticker})...")
        try:
            df = provider.get_spot_prices(ticker, start_str, end_str)
            if df.empty and not use_synthetic:
                logger.warning(f"Live fetch returned empty for {symbol}. Falling back to synthetic provider.")
                syn_p = SyntheticProvider()
                df = syn_p.get_spot_prices(symbol, start_str, end_str)
        except Exception as e:
            logger.warning(f"Error downloading {symbol}: {str(e)}. Using fallback.")
            syn_p = SyntheticProvider()
            df = syn_p.get_spot_prices(symbol, start_str, end_str)

        # Validate
        df_val, val_summary = validator.validate_series(symbol, df)
        quality_summaries.append(val_summary)
        raw_dfs[symbol] = df_val

        # Save raw / processed
        storage.save_dataframe(df_val, f"processed/metals/{symbol.lower()}.parquet")

    # Build Quality Report
    quality_report = validator.build_quality_report(execution_date, quality_summaries)
    storage.save_json(quality_report, f"quality/{execution_date}.json")

    # Extract Key Instruments
    gold_df = raw_dfs.get("GOLD", pd.DataFrame())
    silver_df = raw_dfs.get("SILVER", pd.DataFrame())
    usdinr_df = raw_dfs.get("USDINR", pd.DataFrame())

    # USD/INR latest FX rate
    usdinr_rate = float(usdinr_df["Close"].iloc[-1]) if not usdinr_df.empty and "Close" in usdinr_df.columns else 83.0

    # 2. Technical Analysis
    gold_tech = TechnicalAnalysis.calculate_all(gold_df)
    silver_tech = TechnicalAnalysis.calculate_all(silver_df)

    # 3. Macro Analysis & Regime
    macro_symbols = ["DXY", "US10Y", "US2Y", "OIL_WTI", "SP500", "NIFTY50", "VIX", "INDIA_VIX", "USDINR"]
    macro_dfs = {k: raw_dfs[k] for k in macro_symbols if k in raw_dfs}
    macro_metrics = MacroAnalyzer.calculate_macro_metrics(macro_dfs)
    macro_regime = MacroAnalyzer.classify_macro_regime(macro_metrics)

    # 4. Gold & Silver Specifics & Relative Value Ratio
    gold_analysis = GoldAnalyzer.analyze_gold(gold_df, macro_dfs)
    silver_analysis = SilverAnalyzer.analyze_silver(silver_df, gold_df, macro_dfs)
    ratio_metrics = RatioAnalyzer.calculate_ratio_metrics(gold_df, silver_df)

    # 5. Market Regime
    gold_regime = MarketRegime.classify_metal_regime(gold_tech)
    silver_regime = MarketRegime.classify_metal_regime(silver_tech)

    # 6. Feature Datasets
    gsr_series = ratio_metrics.get("series", None)
    gold_features = FeatureBuilder.build_feature_dataset("GOLD", gold_df, macro_dfs, gsr_series)
    silver_features = FeatureBuilder.build_feature_dataset("SILVER", silver_df, macro_dfs, gsr_series)

    storage.save_dataframe(gold_features, "features/daily/gold_features.parquet")
    storage.save_dataframe(silver_features, "features/daily/silver_features.parquet")

    # 7. Historical Analogue Engine
    analogue_engine = AnalogueEngine()
    gold_analogues = analogue_engine.find_analogues(gold_features, execution_date, top_n=5)
    silver_analogues = analogue_engine.find_analogues(silver_features, execution_date, top_n=5)

    # 8. ML Models & Probabilistic Signals
    forecaster = MetalsForecaster()
    gold_signals = {}
    silver_signals = {}

    for h in [1, 3, 5, 10, 20, 60]:
        g_pred = forecaster.train_and_predict(gold_features, horizon_days=h)
        s_pred = forecaster.train_and_predict(silver_features, horizon_days=h)

        g_stats = gold_analogues.get("forward_horizon_statistics", {})
        s_stats = silver_analogues.get("forward_horizon_statistics", {})

        gold_signals[f"{h}d"] = ProbabilisticEngine.generate_probabilistic_signals(
            h, g_stats, g_pred, gold_df["Close"].pct_change() if not gold_df.empty else pd.Series()
        )
        silver_signals[f"{h}d"] = ProbabilisticEngine.generate_probabilistic_signals(
            h, s_stats, s_pred, silver_df["Close"].pct_change() if not silver_df.empty else pd.Series()
        )

    # 9. Composite Scores
    gold_last_tech = gold_tech.iloc[-1] if not gold_tech.empty else pd.Series()
    silver_last_tech = silver_tech.iloc[-1] if not silver_tech.empty else pd.Series()

    gold_scores = CompositeScorer.calculate_scores(
        gold_last_tech, macro_regime, ratio_metrics, gold_analogues.get("forward_horizon_statistics", {}),
        model_prob=gold_signals.get("5d", {}).get("prob_positive_return", 50.0) / 100.0
    )
    silver_scores = CompositeScorer.calculate_scores(
        silver_last_tech, macro_regime, ratio_metrics, silver_analogues.get("forward_horizon_statistics", {}),
        model_prob=silver_signals.get("5d", {}).get("prob_positive_return", 50.0) / 100.0
    )

    # Calculate INR converted prices
    gold_usd = float(gold_df["Close"].iloc[-1]) if not gold_df.empty else 0.0
    silver_usd = float(silver_df["Close"].iloc[-1]) if not silver_df.empty else 0.0

    # 1 oz = 31.1034768 grams. 10g Gold INR = (gold_usd * usdinr_rate / 31.1034768) * 10
    gold_inr_calc = (gold_usd * usdinr_rate / 31.1034768) * 10.0
    # 1 kg Silver INR = (silver_usd * usdinr_rate / 31.1034768) * 1000.0
    silver_inr_calc = (silver_usd * usdinr_rate / 31.1034768) * 1000.0

    # Package Structured Outputs
    gold_output = {
        "price": gold_usd,
        "inr_price": round(gold_inr_calc, 2),
        "fx_rate_used": usdinr_rate,
        "return_1d_pct": gold_analysis.get("return_1d_pct", 0.0),
        "composite_score": gold_scores["composite_score"],
        "scores": gold_scores,
        "regime": gold_regime,
        "analogues": gold_analogues,
        "signals": gold_signals,
        "research_bias": gold_signals.get("5d", {}).get("research_bias", "Neutral")
    }

    silver_output = {
        "price": silver_usd,
        "inr_price": round(silver_inr_calc, 2),
        "fx_rate_used": usdinr_rate,
        "return_1d_pct": silver_analysis.get("return_1d_pct", 0.0),
        "composite_score": silver_scores["composite_score"],
        "scores": silver_scores,
        "regime": silver_regime,
        "analogues": silver_analogues,
        "signals": silver_signals,
        "research_bias": silver_signals.get("5d", {}).get("research_bias", "Neutral")
    }

    macro_output = {
        "macro_metrics": macro_metrics,
        "macro_regime": macro_regime
    }

    # 10. Alerts Detection
    alert_gen = AlertGenerator()
    alerts = alert_gen.detect_alerts(execution_date, gold_last_tech, silver_last_tech, ratio_metrics, macro_metrics)
    storage.save_json({"date": execution_date, "alerts": alerts}, f"analysis/alerts/{execution_date}.json")

    # 11. Report Generation (JSON and Markdown)
    daily_json_data = ReportGenerator.generate_json_report(
        execution_date, gold_output, silver_output, ratio_metrics, macro_output, quality_report, alerts
    )
    storage.save_json(daily_json_data, f"analysis/daily/{execution_date}.json")

    markdown_content = ReportGenerator.generate_markdown_report(
        execution_date, gold_output, silver_output, ratio_metrics, macro_output, quality_report, alerts
    )

    reports_dir = os.path.join(config["system"].get("reports_dir", "reports"), "daily")
    os.makedirs(reports_dir, exist_ok=True)
    report_file_path = os.path.join(reports_dir, f"{execution_date}.md")
    with open(report_file_path, "w") as f:
        f.write(markdown_content)

    logger.info(f"Daily Research Report saved to {report_file_path}")
    logger.info("--- Pipeline Completed Successfully ---")

if __name__ == "__main__":
    date_arg = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else None
    synthetic_flag = "--synthetic" in sys.argv
    run_pipeline(execution_date=date_arg, use_synthetic=synthetic_flag)
