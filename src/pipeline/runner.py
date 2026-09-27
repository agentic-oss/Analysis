import os
import json
import logging
import datetime
import sys
import pandas as pd
import numpy as np

from src.ingestion.yfinance_provider import YFinanceProvider
from src.ingestion.currency import process_and_align_datasets
from src.validation.validator import DataValidator
from src.storage.storage_manager import StorageManager
from src.metals.metals_analyzer import GoldAnalyzer, SilverAnalyzer
from src.ratios.relative_value import RelativeValueAnalyzer
from src.regimes.regime_detector import MacroRegimeDetector, MarketRegimeDetector
from src.patterns.analogue_engine import HistoricalAnalogueEngine
from src.forecasting.probability_engine import ForwardProbabilityEngine
from src.features.feature_engine import FeatureEngine
from src.scoring.market_scorer import MarketScorer
from src.reporting.alert_generator import AlertGenerator
from src.reporting.report_generator import ReportGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PipelineRunner")

class DailyPipelineRunner:
    """End-to-end daily precious metals research, analysis, and forecasting pipeline."""

    def __init__(self, config_path: str = "config/instruments.json"):
        self.provider = YFinanceProvider(config_path=config_path)
        self.validator = DataValidator()
        self.storage = StorageManager()
        self.gold_analyzer = GoldAnalyzer()
        self.silver_analyzer = SilverAnalyzer()
        self.feature_engine = FeatureEngine()
        self.prob_engine = ForwardProbabilityEngine()
        self.alert_gen = AlertGenerator()
        self.report_gen = ReportGenerator()

        with open(config_path, "r") as f:
            self.instruments = json.load(f)["instruments"]

    def run_pipeline(self, target_date: str = None) -> str:
        logger.info("Starting Daily Precious Metals Research Pipeline execution...")

        # 1. Collect market data for instruments
        data_map = {}
        quality_reports = {}

        for inst in self.instruments:
            sym = inst["symbol"]
            df = self.provider.get_historical_prices(symbol=sym)
            if not df.empty:
                val_df, metrics = self.validator.validate_series(df, sym)
                data_map[sym] = val_df
                quality_reports[sym] = metrics

        # Save data quality report
        today_str = target_date or datetime.datetime.now().strftime("%Y-%m-%d")
        summary_quality = {
            "date": today_str,
            "instruments_count": len(data_map),
            "valid_records": sum(m["valid_records"] for m in quality_reports.values()),
            "warnings": [w for m in quality_reports.values() for w in m["warnings"]],
            "errors": [e for m in quality_reports.values() for e in m["errors"]]
        }
        self.storage.save_quality_report(today_str, summary_quality)

        # 2. Process and align master dataset
        master_df = process_and_align_datasets(data_map)
        if master_df.empty:
            logger.error("Master DataFrame empty. Aborting pipeline.")
            return ""

        # 3. Compute Gold & Silver technical & cross-market correlation indicators
        master_df = self.gold_analyzer.analyze(master_df)
        master_df = self.silver_analyzer.analyze(master_df)

        # 4. Compute Gold/Silver Relative Value metrics
        if "GOLD_close" in master_df.columns and "SILVER_close" in master_df.columns:
            rv_df = RelativeValueAnalyzer.analyze_gold_silver_ratio(master_df["GOLD_close"], master_df["SILVER_close"])
            for c in ["gold_silver_ratio", "ratio_zscore_252d", "ratio_percentile_252d"]:
                if c in rv_df.columns:
                    master_df[c] = rv_df[c]

        # 5. Compute Market & Macro Regimes
        master_df = MarketRegimeDetector.classify_market_regime(master_df, prefix="gold_")
        master_df = MarketRegimeDetector.classify_market_regime(master_df, prefix="silver_")

        # Derive DXY return and VIX for macro regime
        if "DXY_close" in master_df.columns:
            master_df["DXY_close_ret_20d"] = master_df["DXY_close"].pct_change(20)
        if "US10Y_close" in master_df.columns:
            master_df["US10Y_close_chg_20d"] = master_df["US10Y_close"].diff(20)
        if "CRUDE_OIL_close" in master_df.columns:
            master_df["CRUDE_OIL_close_ret_20d"] = master_df["CRUDE_OIL_close"].pct_change(20)
        if "SP500_close" in master_df.columns:
            master_df["SP500_close_ret_20d"] = master_df["SP500_close"].pct_change(20)

        macro_regimes = []
        for i in range(len(master_df)):
            macro_info = MacroRegimeDetector.classify_macro_regime(master_df.iloc[i])
            macro_regimes.append(macro_info["macro_regime"])
        master_df["macro_regime"] = macro_regimes

        # 6. Save processed dataset
        self.storage.save_processed_dataset(master_df, category="metals", name="master_precious_metals")

        # 7. Generate ML Feature store
        events_df = self.provider.get_economic_indicators()
        feature_df = self.feature_engine.build_features(master_df, events_df)
        self.storage.save_processed_dataset(feature_df, category="features", name="daily_features")

        # 8. Target Date Analysis & Forecasting
        if target_date and target_date in master_df["date"].values:
            current_idx = master_df[master_df["date"] == target_date].index[0]
        else:
            current_idx = len(master_df) - 1

        analysis_date = master_df.iloc[current_idx]["date"]

        # Alerts
        alerts = self.alert_gen.check_alerts(master_df, current_idx)

        # Probabilistic Signals & Analogues
        gold_forecast = self.prob_engine.generate_probabilistic_signals(master_df, current_idx, instrument="GOLD")
        silver_forecast = self.prob_engine.generate_probabilistic_signals(master_df, current_idx, instrument="SILVER")

        # Scoring
        gold_scores = MarketScorer.calculate_scores(master_df, current_idx, prefix="gold_")
        silver_scores = MarketScorer.calculate_scores(master_df, current_idx, prefix="silver_")

        # Summaries
        curr_row = master_df.iloc[current_idx]
        gold_summary = {
            "price": float(curr_row.get("GOLD_close", 0.0)),
            "inr_price_10g": float(curr_row.get("GOLD_INR_calc_10g", 0.0)),
            "daily_move_pct": float(curr_row.get("gold_ret_1d", 0.0)) if "gold_ret_1d" in curr_row else 0.0,
            "weekly_move_pct": float(curr_row.get("gold_ret_5d", 0.0)) if "gold_ret_5d" in curr_row else 0.0,
            "trend_regime": curr_row.get("gold_trend_regime", "N/A"),
            "volatility_regime": curr_row.get("gold_volatility_regime", "N/A"),
            "rsi": float(curr_row.get("gold_rsi_14", 50.0)),
            "sma_200": float(curr_row.get("gold_sma_200", 0.0)),
            "high_52w": float(curr_row.get("GOLD_close", 0.0))
        }

        silver_summary = {
            "price": float(curr_row.get("SILVER_close", 0.0)),
            "inr_price_kg": float(curr_row.get("SILVER_INR_calc_kg", 0.0)),
            "daily_move_pct": float(curr_row.get("silver_ret_1d", 0.0)) if "silver_ret_1d" in curr_row else 0.0,
            "weekly_move_pct": float(curr_row.get("silver_ret_5d", 0.0)) if "silver_ret_5d" in curr_row else 0.0,
            "trend_regime": curr_row.get("silver_trend_regime", "N/A"),
            "volatility_regime": curr_row.get("silver_volatility_regime", "N/A")
        }

        ratio_summary = {
            "current_ratio": float(curr_row.get("gold_silver_ratio", 0.0)),
            "zscore_252d": float(curr_row.get("ratio_zscore_252d", 0.0)),
            "percentile_252d": float(curr_row.get("ratio_percentile_252d", 50.0))
        }

        macro_summary = {
            "dxy": float(curr_row.get("DXY_close", 0.0)),
            "us10y": float(curr_row.get("US10Y_close", 0.0)),
            "oil": float(curr_row.get("CRUDE_OIL_close", 0.0)),
            "macro_regime": curr_row.get("macro_regime", "Neutral")
        }

        scores = {"gold_scores": gold_scores, "silver_scores": silver_scores}
        forecast_stats = {
            "gold_10d": gold_forecast.get("horizons", {}).get("10d", {}),
            "silver_10d": silver_forecast.get("horizons", {}).get("10d", {})
        }

        # 9. Generate Daily Reports
        md_p, json_p = self.report_gen.generate_daily_report(
            date_str=str(analysis_date),
            gold_summary=gold_summary,
            silver_summary=silver_summary,
            ratio_summary=ratio_summary,
            macro_summary=macro_summary,
            scores=scores,
            forecast_stats=forecast_stats,
            quality_metrics=summary_quality
        )

        logger.info(f"Pipeline executed successfully for date {analysis_date}. Report: {md_p}")
        return md_p

if __name__ == "__main__":
    date_arg = sys.argv[1] if len(sys.argv) > 1 else None
    runner = DailyPipelineRunner()
    runner.run_pipeline(target_date=date_arg)
