import sys
import os
import json
import yaml
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import pandas as pd

from src.ingestion.provider import YFinanceDataProvider, MockDataProvider
from src.validation.validator import DataValidator
from src.storage.manager import StorageManager, enrich_with_inr_conversions
from src.indicators.technical import calculate_technical_indicators
from src.macro.correlations import analyze_cross_market_relationships
from src.ratios.gold_silver import analyze_gold_silver_ratio
from src.regimes.detector import RegimeClassifier
from src.patterns.analogue import AnalogueEngine
from src.forecasting.probabilistic import ForwardProbabilityEngine
from src.features.builder import FeatureStoreBuilder
from src.scoring.composite import MarketScorer
from src.reporting.daily_report import ResearchReportGenerator, AlertEngine
from src.evaluation.performance import ForecastEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PipelineRunner")

class MetalsPipelineRunner:
    """End-to-end Daily Precious Metals Research Pipeline Runner."""

    def __init__(self, config_path: str = "config/config.yaml", instruments_path: str = "config/instruments.json"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        with open(instruments_path, "r") as f:
            self.instruments = json.load(f)["instruments"]

        self.storage = StorageManager()
        self.validator = DataValidator(jump_threshold_pct=self.config.get("alerts", {}).get("price_jump_pct", 3.5))

    def run_daily_pipeline(self, target_date_str: str, use_mock: bool = False) -> Dict[str, Any]:
        """Execute all steps: Ingestion -> Validation -> Storage -> Indicators -> Macro -> Ratios -> Regimes -> Analogue -> Forecasting -> Scoring -> Reporting."""
        logger.info(f"=== Starting Daily Pipeline Execution for Date: {target_date_str} ===")

        # Date range for ingestion (last 2 years up to target_date)
        dt_end = datetime.strptime(target_date_str, "%Y-%m-%d") + timedelta(days=1)
        dt_start = dt_end - timedelta(days=730)
        start_str = dt_start.strftime("%Y-%m-%d")
        end_str = dt_end.strftime("%Y-%m-%d")

        # 1. Data Ingestion
        if use_mock:
            provider = MockDataProvider()
        else:
            provider = YFinanceDataProvider(
                retries=self.config.get("system", {}).get("retries", 3),
                timeout=self.config.get("system", {}).get("timeout_seconds", 10)
            )

        raw_datasets = provider.fetch_all_instruments(self.instruments, start_str, end_str)

        # 2. Data Validation & Storage
        quality_reports = {}
        processed_datasets = {}

        for symbol, df_raw in raw_datasets.items():
            if df_raw.empty:
                continue

            # Save raw
            self.storage.save_raw(target_date_str, symbol, df_raw)

            # Validate
            df_validated, q_report = self.validator.validate_dataset(symbol, df_raw)
            quality_reports[symbol] = q_report

            # Calculate technical indicators
            df_ind = calculate_technical_indicators(df_validated, self.config)

            # Find category
            inst_meta = next((i for i in self.instruments if i["symbol"] == symbol), {})
            cat = inst_meta.get("category", "metals")

            # Save processed
            self.storage.save_processed(cat, symbol, df_ind)
            processed_datasets[symbol] = df_ind

        # Quality Report Summary
        q_report_path = self.validator.generate_quality_report(target_date_str, quality_reports)
        logger.info(f"Data quality report generated: {q_report_path}")

        # INR Currency Conversion Enrichment
        gold_df = processed_datasets.get("GOLD")
        silver_df = processed_datasets.get("SILVER")
        usdinr_df = processed_datasets.get("USDINR")

        if gold_df is not None and usdinr_df is not None:
            factor_10g = self.config.get("units", {}).get("gold_oz_to_10g", 3.11034768 / 31.1034768)
            gold_df = enrich_with_inr_conversions(gold_df, usdinr_df, factor_10g)
            processed_datasets["GOLD"] = gold_df

        if silver_df is not None and usdinr_df is not None:
            factor_kg = self.config.get("units", {}).get("silver_oz_to_kg", 0.0311034768)
            silver_df = enrich_with_inr_conversions(silver_df, usdinr_df, factor_kg)
            processed_datasets["SILVER"] = silver_df

        # 3. Ratio Analysis
        ratio_df = pd.DataFrame()
        if gold_df is not None and silver_df is not None:
            ratio_df = analyze_gold_silver_ratio(gold_df, silver_df)

        # 4. Regime Classification
        macro_dict = RegimeClassifier.classify_macro_regime(
            dxy_df=processed_datasets.get("DXY"),
            us10y_df=processed_datasets.get("US10Y"),
            tip_df=processed_datasets.get("TIP"),
            oil_df=processed_datasets.get("CRUDE_OIL"),
            sp500_df=processed_datasets.get("SP500"),
            vix_df=processed_datasets.get("VIX")
        )

        gold_regime = RegimeClassifier.classify_market_regime(gold_df) if gold_df is not None else {}
        silver_regime = RegimeClassifier.classify_market_regime(silver_df) if silver_df is not None else {}

        # 5. Build Independent Feature Datasets for Gold & Silver
        gold_feat_df = FeatureStoreBuilder.build_daily_features(processed_datasets, ratio_df, target_symbol="GOLD", horizons=[1, 3, 5, 10, 20, 60])
        silver_feat_df = FeatureStoreBuilder.build_daily_features(processed_datasets, ratio_df, target_symbol="SILVER", horizons=[1, 3, 5, 10, 20, 60])

        if not gold_feat_df.empty:
            self.storage.save_features(gold_feat_df, filename=f"features_gold_{target_date_str}.parquet")
        if not silver_feat_df.empty:
            self.storage.save_features(silver_feat_df, filename=f"features_silver_{target_date_str}.parquet")

        # 6. Analogue Engine & Forecasting Signals
        analogue_engine = AnalogueEngine(top_k=self.config.get("forecasting", {}).get("analogue_top_k", 20))
        weights = self.config.get("scores", {}).get("weights", {})

        # Gold Analogue & Score
        gold_signals = {}
        gold_score_res = {}
        gold_analogue_res = {}
        if not gold_feat_df.empty:
            curr_row_g = gold_feat_df.iloc[-1]
            hist_slice_g = gold_feat_df.iloc[:-1] if len(gold_feat_df) > 1 else gold_feat_df
            gold_analogue_res = analogue_engine.find_analogues(hist_slice_g, curr_row_g, horizons=[1, 3, 5, 10, 20, 60])
            gold_signals = ForwardProbabilityEngine.generate_probabilistic_signals(gold_analogue_res.get("forward_statistics", {}))
            gold_score_res = MarketScorer.calculate_scores(curr_row_g, macro_dict, gold_analogue_res, weights)

        # Silver Analogue & Score
        silver_signals = {}
        silver_score_res = {}
        silver_analogue_res = {}
        if not silver_feat_df.empty:
            curr_row_s = silver_feat_df.iloc[-1]
            hist_slice_s = silver_feat_df.iloc[:-1] if len(silver_feat_df) > 1 else silver_feat_df
            silver_analogue_res = analogue_engine.find_analogues(hist_slice_s, curr_row_s, horizons=[1, 3, 5, 10, 20, 60])
            silver_signals = ForwardProbabilityEngine.generate_probabilistic_signals(silver_analogue_res.get("forward_statistics", {}))
            silver_score_res = MarketScorer.calculate_scores(curr_row_s, macro_dict, silver_analogue_res, weights)

        # 7. Format Daily Analysis Payload
        analysis_payload = {
            "date": target_date_str,
            "gold": {
                "price": float(gold_df["close"].iloc[-1]) if gold_df is not None and not gold_df.empty else 0.0,
                "return_1d_pct": float(gold_df["close"].pct_change().iloc[-1]*100) if gold_df is not None and len(gold_df) > 1 else 0.0,
                "return_5d_pct": float(gold_df["close"].pct_change(5).iloc[-1]*100) if gold_df is not None and len(gold_df) > 5 else 0.0,
                "return_20d_pct": float(gold_df["close"].pct_change(20).iloc[-1]*100) if gold_df is not None and len(gold_df) > 20 else 0.0,
                "trend_regime": gold_regime.get("trend_regime", "N/A"),
                "volatility_regime": gold_regime.get("volatility_regime", "N/A"),
                "rsi_14": float(gold_df["rsi_14"].iloc[-1]) if gold_df is not None and "rsi_14" in gold_df.columns else 50.0,
                "macd": float(gold_df["macd"].iloc[-1]) if gold_df is not None and "macd" in gold_df.columns else 0.0,
                "sma_50": float(gold_df["sma_50"].iloc[-1]) if gold_df is not None and "sma_50" in gold_df.columns else 0.0,
                "sma_200": float(gold_df["sma_200"].iloc[-1]) if gold_df is not None and "sma_200" in gold_df.columns else 0.0,
                "bb_upper": float(gold_df["bb_upper"].iloc[-1]) if gold_df is not None and "bb_upper" in gold_df.columns else 0.0,
                "dist_sma_200_pct": float(gold_df["dist_sma_200"].iloc[-1]*100) if gold_df is not None and "dist_sma_200" in gold_df.columns else 0.0,
                "composite_score": gold_score_res.get("composite_score", 50.0),
                "research_bias": "Bullish" if gold_score_res.get("composite_score", 50.0) >= 55 else ("Bearish" if gold_score_res.get("composite_score", 50.0) <= 45 else "Neutral"),
                "signals": gold_signals,
                "analogue_sample_count": len(gold_analogue_res.get("analogues", []))
            },
            "silver": {
                "price": float(silver_df["close"].iloc[-1]) if silver_df is not None and not silver_df.empty else 0.0,
                "return_1d_pct": float(silver_df["close"].pct_change().iloc[-1]*100) if silver_df is not None and len(silver_df) > 1 else 0.0,
                "return_5d_pct": float(silver_df["close"].pct_change(5).iloc[-1]*100) if silver_df is not None and len(silver_df) > 5 else 0.0,
                "return_20d_pct": float(silver_df["close"].pct_change(20).iloc[-1]*100) if silver_df is not None and len(silver_df) > 20 else 0.0,
                "trend_regime": silver_regime.get("trend_regime", "N/A"),
                "volatility_regime": silver_regime.get("volatility_regime", "N/A"),
                "rsi_14": float(silver_df["rsi_14"].iloc[-1]) if silver_df is not None and "rsi_14" in silver_df.columns else 50.0,
                "macd": float(silver_df["macd"].iloc[-1]) if silver_df is not None and "macd" in silver_df.columns else 0.0,
                "composite_score": silver_score_res.get("composite_score", 50.0),
                "research_bias": "Bullish" if silver_score_res.get("composite_score", 50.0) >= 55 else ("Bearish" if silver_score_res.get("composite_score", 50.0) <= 45 else "Neutral"),
                "signals": silver_signals,
                "analogue_sample_count": len(silver_analogue_res.get("analogues", []))
            },
            "gold_silver_ratio": {
                "current_ratio": float(ratio_df["gold_silver_ratio"].iloc[-1]) if not ratio_df.empty else 0.0,
                "zscore": float(ratio_df["ratio_zscore"].iloc[-1]) if not ratio_df.empty else 0.0,
                "percentile_252": float(ratio_df["ratio_percentile_252"].iloc[-1]) if not ratio_df.empty else 50.0
            },
            "macro": macro_dict,
            "data_quality": {
                "total_instruments_validated": len(quality_reports),
                "total_errors": sum(len(r["errors"]) for r in quality_reports.values()),
                "total_warnings": sum(len(r["warnings"]) for r in quality_reports.values())
            }
        }

        # Save machine-readable analysis JSON
        self.storage.save_daily_analysis_json(target_date_str, analysis_payload)

        # Detect and save alerts
        alerts = AlertEngine.detect_alerts(
            target_date_str,
            gold_df.iloc[-1] if gold_df is not None and not gold_df.empty else None,
            silver_df.iloc[-1] if silver_df is not None and not silver_df.empty else None,
            ratio_df.iloc[-1] if not ratio_df.empty else None,
            macro_dict,
            self.config
        )
        self.storage.save_daily_alerts_json(target_date_str, alerts)

        # Generate Markdown Report
        report_md_path = ResearchReportGenerator.generate_markdown_report(target_date_str, analysis_payload)
        logger.info(f"Markdown report generated: {report_md_path}")

        logger.info(f"=== Pipeline completed successfully for {target_date_str} ===")
        return analysis_payload

if __name__ == "__main__":
    date_arg = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    use_mock_flag = "--mock" in sys.argv or "--offline" in sys.argv
    runner = MetalsPipelineRunner()
    runner.run_daily_pipeline(date_arg, use_mock=use_mock_flag)
