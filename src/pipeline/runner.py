"""
Daily Gold & Silver Pipeline Runner.
Executes end-to-end collection, validation, indicators, macro regimes,
analogue engine, feature store, models, backtesting, report generation, and alert dispatch.
"""

from datetime import datetime, date, timedelta, timezone
import json
import logging
import os
import sys
from typing import Dict, Any, List, Optional
import pandas as pd
import yaml

from src.ingestion.ingest import DataIngestionService
from src.ingestion.provider import YFinanceProvider, SyntheticProvider
from src.validation.validator import DataValidator
from src.storage.storage import StorageManager
from src.indicators.technical import TechnicalAnalysisEngine
from src.metals.indian_market import IndianMarketConverter
from src.ratios.ratio_engine import RatioAnalysisEngine, CrossMarketCorrelationEngine
from src.regimes.macro_regime import MacroRegimeClassifier, MarketRegimeClassifier
from src.patterns.analogue_engine import HistoricalAnalogueEngine
from src.forecasting.probabilistic import ProbabilisticSignalEngine
from src.features.feature_store import FeatureStoreEngine
from src.models.forecasting_models import MetalsForecastingModel, BacktestEngine, ForecastEvaluationTracker
from src.scoring.scoring import ScoringEngine, AlertEngine
from src.reporting.report_generator import ReportGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PipelineRunner")


class DailyPipelineRunner:
    def __init__(self, config_path: str = "config/config.yaml", instruments_path: str = "config/instruments.json"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        with open(instruments_path, "r") as f:
            self.instruments_cfg = json.load(f)

        self.storage = StorageManager(base_dir=self.config.get("system", {}).get("data_dir", "data"))
        self.validator = DataValidator()

        # Ingestion service with fallback
        primary_provider = YFinanceProvider(
            retries=self.config.get("ingestion", {}).get("retry_attempts", 3),
            backoff_sec=self.config.get("ingestion", {}).get("retry_backoff_sec", 2),
        )
        fallback_provider = SyntheticProvider()
        self.ingestion_service = DataIngestionService(primary_provider, fallback_provider)

        self.tech_engine = TechnicalAnalysisEngine()
        self.indian_converter = IndianMarketConverter(
            custom_duty_pct=self.config.get("conversions", {}).get("custom_duty_gold_pct", 0.06)
        )
        self.ratio_engine = RatioAnalysisEngine()
        self.macro_classifier = MacroRegimeClassifier()
        self.market_classifier = MarketRegimeClassifier()
        self.analogue_engine = HistoricalAnalogueEngine()
        self.signal_engine = ProbabilisticSignalEngine()
        self.feature_engine = FeatureStoreEngine()
        self.scoring_engine = ScoringEngine(weights=self.config.get("scoring_weights"))
        self.alert_engine = AlertEngine()
        self.report_generator = ReportGenerator()

    def run_pipeline(self, target_date_str: Optional[str] = None) -> Dict[str, Any]:
        """Runs daily pipeline for target_date_str (YYYY-MM-DD)."""
        if not target_date_str:
            target_date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        logger.info(f"Starting Precious Metals Analysis Pipeline for target date: {target_date_str}")

        start_date_dt = datetime.strptime(target_date_str, "%Y-%m-%d") - timedelta(
            days=self.config.get("ingestion", {}).get("lookback_days", 365)
        )
        start_date_str = start_date_dt.strftime("%Y-%m-%d")

        # 1. Fetch & Validate Instruments
        collected_dfs = {}
        validation_summaries = []

        for inst in self.instruments_cfg.get("instruments", []):
            if not inst.get("enabled", True):
                continue
            symbol = inst["symbol"]
            logger.info(f"Fetching data for {symbol}...")
            df = self.ingestion_service.fetch_instrument_history(inst, start_date_str, target_date_str)
            v_df, summary = self.validator.validate_dataframe(df, symbol)
            collected_dfs[symbol] = v_df
            validation_summaries.append(summary)

            # Save raw dataset
            raw_dir = self.storage.get_raw_dir(target_date_str)
            self.storage.save_dataframe(v_df, os.path.join(raw_dir, f"{symbol}.parquet"))

        # Save quality report
        quality_file = os.path.join(self.storage.base_dir, "quality", f"{target_date_str}.json")
        self.validator.generate_quality_report(validation_summaries, target_date_str, quality_file)

        # 2. Technical Analysis for Gold & Silver
        gold_df = self.tech_engine.compute_indicators(collected_dfs.get("GOLD", collected_dfs.get("GOLD_FUTURES")))
        silver_df = self.tech_engine.compute_indicators(collected_dfs.get("SILVER", collected_dfs.get("SILVER_FUTURES")))

        # Save processed metals
        proc_metals_dir = self.storage.get_processed_dir("metals")
        self.storage.save_dataframe(gold_df, os.path.join(proc_metals_dir, "gold_processed.parquet"))
        self.storage.save_dataframe(silver_df, os.path.join(proc_metals_dir, "silver_processed.parquet"))

        # 3. Indian Market Price Conversions
        usdinr_df = collected_dfs.get("USDINR", pd.DataFrame())
        if not usdinr_df.empty and not gold_df.empty and not silver_df.empty:
            indian_df = self.indian_converter.process_indian_metals_dataset(gold_df, silver_df, usdinr_df)
            self.storage.save_dataframe(indian_df, os.path.join(proc_metals_dir, "indian_metals_processed.parquet"))

        # 4. Gold/Silver Ratio Metrics
        ratio_df = self.ratio_engine.calculate_ratio_metrics(gold_df, silver_df)

        # 5. Build ML Daily Features
        daily_features_df = self.feature_engine.build_daily_features(gold_df)
        features_dir = os.path.join(self.storage.base_dir, "features", "daily")
        self.storage.save_dataframe(daily_features_df, os.path.join(features_dir, f"gold_features_{target_date_str}.parquet"))

        # Extract latest metrics for current_idx
        current_idx = len(gold_df) - 1
        latest_gold = gold_df.iloc[current_idx].to_dict() if current_idx >= 0 else {}
        latest_silver = silver_df.iloc[-1].to_dict() if not silver_df.empty else {}
        latest_ratio = ratio_df.iloc[-1].to_dict() if not ratio_df.empty else {}

        # 6. Macro & Market Regimes
        macro_regime = self.macro_classifier.classify_macro_regime(
            dxy_change_20d=collected_dfs.get("DXY", pd.DataFrame())["close"].pct_change(20).iloc[-1] if "DXY" in collected_dfs and len(collected_dfs["DXY"]) >= 20 else 0.0,
            yield_10y_change_20d=collected_dfs.get("US10Y", pd.DataFrame())["close"].diff(20).iloc[-1] if "US10Y" in collected_dfs and len(collected_dfs["US10Y"]) >= 20 else 0.0,
            real_yield_change_20d=collected_dfs.get("REAL_YIELD", pd.DataFrame())["close"].diff(20).iloc[-1] if "REAL_YIELD" in collected_dfs and len(collected_dfs["REAL_YIELD"]) >= 20 else 0.0,
            vix_level=collected_dfs.get("VIX", pd.DataFrame())["close"].iloc[-1] if "VIX" in collected_dfs else 15.0,
            oil_return_20d=collected_dfs.get("CRUDE_OIL", pd.DataFrame())["close"].pct_change(20).iloc[-1] if "CRUDE_OIL" in collected_dfs and len(collected_dfs["CRUDE_OIL"]) >= 20 else 0.0,
            sp500_return_20d=collected_dfs.get("SP500", pd.DataFrame())["close"].pct_change(20).iloc[-1] if "SP500" in collected_dfs and len(collected_dfs["SP500"]) >= 20 else 0.0,
        )

        gold_market_regime = self.market_classifier.classify_market_regime(
            close=latest_gold.get("close", 0.0),
            sma_20=latest_gold.get("sma_20", 0.0),
            sma_50=latest_gold.get("sma_50", 0.0),
            sma_200=latest_gold.get("sma_200", 0.0),
            rsi=latest_gold.get("rsi_14", 50.0),
            volatility_20d=latest_gold.get("volatility_20d", 0.15),
        )

        # 7. Historical Analogue Search
        gold_analogues = self.analogue_engine.find_analogues(daily_features_df, current_idx)

        # 8. Scores & Probabilistic Forecast Signals
        scores = self.scoring_engine.compute_composite_scores(
            tech_indicators=latest_gold,
            macro_indicators=macro_regime.get("components", {}),
            ratio_metrics=latest_ratio,
            analogue_stats=gold_analogues.get("forward_stats", {}),
        )

        gold_signals = self.signal_engine.generate_probabilistic_signals(
            analogue_stats=gold_analogues.get("forward_stats", {}),
            technical_score=scores["technical_score"],
            macro_score=scores["macro_score"],
            model_forecasts={},
        )

        # 9. Alerts Detection
        alerts = self.alert_engine.detect_alerts(
            date_str=target_date_str,
            gold_data=latest_gold,
            silver_data=latest_silver,
            ratio_data=latest_ratio,
            macro_data=macro_regime.get("components", {}),
        )

        # Save alerts
        alerts_file = os.path.join(self.storage.base_dir, "analysis", "alerts", f"{target_date_str}.json")
        self.storage.save_json({"date": target_date_str, "alerts": alerts}, alerts_file)

        # 10. Report Generation
        overall_quality_summary = {
            "valid_records": sum(s["valid_records"] for s in validation_summaries),
            "suspicious_records": sum(s["suspicious_records"] for s in validation_summaries),
            "invalid_records": sum(s["invalid_records"] for s in validation_summaries),
        }

        report_data = {
            "gold": {
                "price_usd": latest_gold.get("close", 0.0),
                "price_inr_10g": self.indian_converter.convert_gold_usd_to_inr_10g(latest_gold.get("close", 0.0), 83.0),
                "return_1d_pct": (latest_gold.get("return_1d", 0.0) or 0.0) * 100.0,
                "return_5d_pct": (latest_gold.get("return_5d", 0.0) or 0.0) * 100.0,
                "return_20d_pct": (latest_gold.get("return_20d", 0.0) or 0.0) * 100.0,
                "rsi_14": latest_gold.get("rsi_14", 50.0),
                "macd_hist": latest_gold.get("macd_hist", 0.0),
                "dist_sma_200_pct": latest_gold.get("dist_sma_200_pct", 0.0),
                "atr": latest_gold.get("atr", 0.0),
                "trend_regime": gold_market_regime.get("trend_regime"),
                "volatility_regime": gold_market_regime.get("volatility_regime"),
            },
            "silver": {
                "price_usd": latest_silver.get("close", 0.0),
                "price_inr_kg": self.indian_converter.convert_silver_usd_to_inr_kg(latest_silver.get("close", 0.0), 83.0),
                "return_1d_pct": (latest_silver.get("return_1d", 0.0) or 0.0) * 100.0,
                "return_5d_pct": (latest_silver.get("return_5d", 0.0) or 0.0) * 100.0,
                "return_20d_pct": (latest_silver.get("return_20d", 0.0) or 0.0) * 100.0,
                "rsi_14": latest_silver.get("rsi_14", 50.0),
                "macd_hist": latest_silver.get("macd_hist", 0.0),
                "dist_sma_200_pct": latest_silver.get("dist_sma_200_pct", 0.0),
                "atr": latest_silver.get("atr", 0.0),
            },
            "ratio": {
                "gold_silver_ratio": latest_ratio.get("gold_silver_ratio", 80.0),
                "ratio_percentile_252": latest_ratio.get("ratio_percentile_252", 50.0),
                "ratio_zscore": latest_ratio.get("ratio_zscore", 0.0),
                "return_spread_1d_pct": (latest_ratio.get("return_spread_1d", 0.0) or 0.0) * 100.0,
            },
            "macro": macro_regime,
            "scores": scores,
            "analogues": {"gold": gold_analogues},
            "signals": {"gold": gold_signals},
            "alerts": alerts,
            "data_quality": overall_quality_summary,
        }

        # Generate Markdown & JSON outputs
        markdown_text = self.report_generator.generate_markdown_report(target_date_str, report_data)
        json_dict = self.report_generator.generate_json_report(target_date_str, report_data)

        md_path = os.path.join(self.config.get("system", {}).get("reports_dir", "reports"), "daily", f"{target_date_str}.md")
        json_path = os.path.join(self.storage.base_dir, "analysis", "daily", f"{target_date_str}.json")

        os.makedirs(os.path.dirname(md_path), exist_ok=True)
        with open(md_path, "w") as f:
            f.write(markdown_text)

        self.storage.save_json(json_dict, json_path)

        logger.info(f"Pipeline execution completed successfully for {target_date_str}!")
        logger.info(f"Markdown report generated: {md_path}")
        logger.info(f"JSON analysis generated: {json_path}")

        return report_data


if __name__ == "__main__":
    target_dt = sys.argv[1] if len(sys.argv) > 1 else None
    runner = DailyPipelineRunner()
    runner.run_pipeline(target_dt)
