"""
Master Daily Execution Pipeline Runner.
Orchestrates Ingestion -> Validation -> Storage -> Indicators -> Macro -> Regimes -> Features -> Scoring -> Forecasting -> Reporting -> Alerts.
"""
import sys
import json
import yaml
import datetime
import logging
from pathlib import Path

from src.ingestion.yfinance_provider import YFinanceProvider
from src.validation.validator import DataValidator
from src.storage.repository import DataRepository
from src.indicators.technical import TechnicalIndicators
from src.metals.gold import GoldAnalyzer, SilverAnalyzer
from src.ratios.gold_silver_ratio import GoldSilverRatio
from src.macro.macro_analysis import MacroAnalyzer
from src.regimes.macro_regimes import MacroRegimeDetector
from src.regimes.market_regimes import MarketRegimeDetector
from src.patterns.analogues import AnalogueEngine
from src.features.builder import FeatureBuilder
from src.scoring.composite_score import CompositeScorer
from src.forecasting.probabilistic import ProbabilisticForecaster
from src.reporting.daily_report import DailyReportGenerator
from src.reporting.alerts import AlertEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(target_date_str: str = None):
    if target_date_str is None:
        target_date_str = datetime.datetime.now().strftime("%Y-%m-%d")

    logger.info(f"Starting Precious Metals Research Pipeline for date: {target_date_str}")

    # Load configs
    with open("config/instruments.json") as f:
        instruments_cfg = json.load(f)["instruments"]

    with open("config/config.yaml") as f:
        config_yaml = yaml.safe_load(f)

    repo = DataRepository(base_dir=config_yaml["system"]["data_dir"])
    provider = YFinanceProvider()
    validator = DataValidator()

    # 1. Fetch & Store Ingested Data
    data_dict = provider.fetch_all_instruments_data(
        instruments_cfg, start_date=config_yaml["ingestion"]["historical_start_date"], end_date=target_date_str
    )

    quality_summaries = []
    processed_dfs = {}

    for sym, df in data_dict.items():
        if df.empty:
            continue
        val_df, q_summary = validator.validate_dataframe(df, sym)
        quality_summaries.append(q_summary)

        # Calculate Technical Indicators
        df_ind = TechnicalIndicators.add_all_indicators(val_df)
        processed_dfs[sym] = df_ind
        repo.save_processed_data(df_ind, sym)

    # Save Quality Report
    quality_report = validator.create_daily_quality_report(target_date_str, quality_summaries)
    repo.save_quality_report(quality_report, target_date_str)

    # 2. Extract Key Datasets
    gold_df = processed_dfs.get("GOLD", None)
    silver_df = processed_dfs.get("SILVER", None)
    usdinr_df = processed_dfs.get("USDINR", None)

    usd_inr_rate = float(usdinr_df["close"].iloc[-1]) if usdinr_df is not None and not usdinr_df.empty else 83.0

    # Metal Analytics
    gold_analysis = GoldAnalyzer.analyze_gold(gold_df, usd_inr_rate=usd_inr_rate) if gold_df is not None else {}
    silver_analysis = SilverAnalyzer.analyze_silver(silver_df, usd_inr_rate=usd_inr_rate) if silver_df is not None else {}

    # Gold/Silver Ratio Analytics
    ratio_df = GoldSilverRatio.calculate_ratio(gold_df, silver_df) if gold_df is not None and silver_df is not None else None
    ratio_analysis = GoldSilverRatio.analyze_latest_ratio(ratio_df) if ratio_df is not None else {}

    # Macro & Regimes
    macro_keys = ["DXY", "US10Y", "CRUDE_OIL", "SP500", "VIX"]
    macro_dfs = {k: processed_dfs[k] for k in macro_keys if k in processed_dfs}

    macro_regime = MacroRegimeDetector.detect_macro_regime(macro_dfs)

    market_regimes = {}
    if gold_df is not None:
        market_regimes["GOLD"] = MarketRegimeDetector.detect_market_regime(gold_df, "GOLD")
    if silver_df is not None:
        market_regimes["SILVER"] = MarketRegimeDetector.detect_market_regime(silver_df, "SILVER")

    # Feature Store & Target Generation
    gold_features = FeatureBuilder.build_daily_features(gold_df, "GOLD", macro_dfs, ratio_df) if gold_df is not None else None
    silver_features = FeatureBuilder.build_daily_features(silver_df, "SILVER", macro_dfs, ratio_df) if silver_df is not None else None

    if gold_features is not None:
        repo.save_daily_features(gold_features, f"GOLD_{target_date_str}")

    # Historical Analogue Search
    gold_analogues = AnalogueEngine.find_analogues(gold_features, gold_features.iloc[-1]) if gold_features is not None and not gold_features.empty else {}
    silver_analogues = AnalogueEngine.find_analogues(silver_features, silver_features.iloc[-1]) if silver_features is not None and not silver_features.empty else {}

    # Composite Scoring
    gold_row = gold_df.iloc[-1].to_dict() if gold_df is not None else {}
    gold_scores = CompositeScorer.calculate_scores(gold_row, macro_regime, ratio_analysis, gold_analogues)
    silver_row = silver_df.iloc[-1].to_dict() if silver_df is not None else {}
    silver_scores = CompositeScorer.calculate_scores(silver_row, macro_regime, ratio_analysis, silver_analogues)

    # Probabilistic Research Signals
    gold_forecast = ProbabilisticForecaster.generate_probabilistic_signal("GOLD", gold_analogues.get("forward_statistics", {}), gold_scores)
    silver_forecast = ProbabilisticForecaster.generate_probabilistic_signal("SILVER", silver_analogues.get("forward_statistics", {}), silver_scores)

    # Save Machine-Readable JSON Analysis
    daily_json = {
        "date": target_date_str,
        "gold": {
            "analysis": gold_analysis,
            "scores": gold_scores,
            "forecast": gold_forecast,
        },
        "silver": {
            "analysis": silver_analysis,
            "scores": silver_scores,
            "forecast": silver_forecast,
        },
        "gold_silver_ratio": ratio_analysis,
        "macro_regime": macro_regime,
        "market_regimes": market_regimes,
        "data_quality": quality_report,
    }
    repo.save_daily_analysis(daily_json, target_date_str)

    # Generate Alerts
    alerts = AlertEngine.detect_alerts(target_date_str, gold_analysis, silver_analysis, ratio_analysis, macro_regime)
    repo.save_alerts(alerts, target_date_str)

    # Generate Markdown Report
    report_file = DailyReportGenerator.generate_markdown_report(
        target_date_str,
        gold_analysis,
        silver_analysis,
        ratio_analysis,
        macro_regime,
        market_regimes,
        gold_forecast,
        silver_forecast,
        quality_report,
    )

    logger.info(f"Pipeline completed successfully. Daily report generated at: {report_file}")


if __name__ == "__main__":
    dt_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(dt_arg)
