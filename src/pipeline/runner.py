"""
Pipeline runner orchestrating the daily end-to-end precious metals analysis pipeline.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from src.ingestion.provider import YFinanceDataProvider, IndianPriceConverter
from src.validation.validator import DataValidator
from src.storage.storage import DataStorage
from src.indicators.technical import TechnicalIndicators
from src.metals.gold import GoldAnalysis
from src.metals.silver import SilverAnalysis
from src.ratios.relative_value import RelativeValueAnalysis
from src.regimes.macro import MacroRegimeDetector
from src.regimes.market import MarketRegimeDetector
from src.patterns.analogues import HistoricalAnalogueEngine
from src.features.builder import FeatureBuilder
from src.scoring.composite import CompositeScorer
from src.models.baseline import MetalsForecastingModels
from src.forecasting.signals import ProbabilisticSignalEngine
from src.backtesting.engine import StrategyBacktester
from src.evaluation.tracker import ForecastEvaluator
from src.scoring.alerts import AlertEngine
from src.reporting.generator import ReportGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class PipelineRunner:
    """Executes the daily precious metals research pipeline."""

    def __init__(self, config_path: str = "config/config.yaml", instruments_path: str = "config/instruments.json"):
        self.storage = DataStorage()
        self.provider = YFinanceDataProvider()
        self.validator = DataValidator()

        with open(instruments_path, "r") as f:
            self.instruments = json.load(f).get("instruments", [])

    def run_daily_pipeline(self, target_date: str = None) -> Dict[str, Any]:
        """Runs the entire pipeline for target_date (defaults to today YYYY-MM-DD)."""
        if not target_date:
            target_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"=== Starting Precious Metals Pipeline for {target_date} ===")

        # 1. Fetch historical raw market data
        raw_datasets = {}
        audit_reports = {}

        for inst in self.instruments:
            symbol = inst["symbol"]
            ticker = inst["ticker"]
            if not inst.get("enabled", True):
                continue

            try:
                df = self.provider.get_historical_prices(ticker, period="2y", interval="1d")
                df_valid, audit = self.validator.validate_series(df, symbol)
                raw_datasets[symbol] = df_valid
                audit_reports[symbol] = audit
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol} ({ticker}): {str(e)}")

        if "GOLD" not in raw_datasets or raw_datasets["GOLD"].empty:
            logger.error("Gold market data missing or invalid. Aborting pipeline run.")
            return {"status": "failed", "reason": "Missing gold data"}

        # Quality audit report save
        self.storage.save_json({
            "date": target_date,
            "audit_reports": audit_reports,
            "total_instruments": len(raw_datasets),
        }, f"quality/{target_date}")

        # 2. Process Technical Indicators
        processed_dfs = {}
        for symbol, df in raw_datasets.items():
            processed_df = TechnicalIndicators.calculate_all(df)
            processed_dfs[symbol] = processed_df
            self.storage.save_dataframe(processed_df, f"processed/{symbol.lower()}_historical")

        gold_df = processed_dfs["GOLD"]
        silver_df = processed_dfs.get("SILVER", pd.DataFrame())
        usdinr_df = processed_dfs.get("USDINR", pd.DataFrame())

        # 3. Currency Conversion & Indian Market Prices
        usdinr_map = usdinr_df.set_index("date")["close"].to_dict() if not usdinr_df.empty else {}

        gold_df["usdinr"] = gold_df["date"].map(usdinr_map).ffill().fillna(83.0)
        gold_df["inr_price_10g"] = gold_df.apply(
            lambda r: IndianPriceConverter.convert_usd_oz_to_inr_10g(r["close"], r["usdinr"]), axis=1
        )

        if not silver_df.empty:
            silver_df["usdinr"] = silver_df["date"].map(usdinr_map).ffill().fillna(83.0)
            silver_df["inr_price_kg"] = silver_df.apply(
                lambda r: IndianPriceConverter.convert_usd_oz_to_inr_kg(r["close"], r["usdinr"]), axis=1
            )

        # 4. Multi-Factor Correlations & Relative Value
        gold_analysis = GoldAnalysis()
        macro_dict = {
            "dxy": processed_dfs.get("DXY", pd.DataFrame()),
            "us10y": processed_dfs.get("US10Y", pd.DataFrame()),
            "oil": processed_dfs.get("CRUDE_OIL", pd.DataFrame()),
            "sp500": processed_dfs.get("SP500", pd.DataFrame()),
            "vix": processed_dfs.get("VIX", pd.DataFrame()),
        }
        gold_corrs = gold_analysis.calculate_rolling_correlations(gold_df, macro_dict)

        rv_analysis = RelativeValueAnalysis()
        rv_df = rv_analysis.analyze_ratio(gold_df, silver_df)

        # 5. Feature Engineering
        features_df = FeatureBuilder.build_daily_features(gold_df, silver_df, macro_dict)

        # 6. Regimes
        features_df = MacroRegimeDetector.classify_df(features_df)
        gold_df = MarketRegimeDetector.classify_instrument(gold_df)
        market_regime_map = gold_df.set_index("date")["market_regime"].to_dict()
        features_df["market_regime"] = features_df["date"].map(market_regime_map).fillna("Neutral / Medium Volatility")

        self.storage.save_dataframe(features_df, "features/daily_features")

        # Select feature row for target_date
        if target_date not in features_df["date"].values:
            target_idx = len(features_df) - 1
            curr_target_date = features_df.loc[target_idx, "date"]
        else:
            target_idx = features_df[features_df["date"] == target_date].index[0]
            curr_target_date = target_date

        feature_row = features_df.iloc[target_idx]

        # 7. Historical Analogue Search
        analogue_engine = HistoricalAnalogueEngine()
        analogue_results = analogue_engine.find_analogues(features_df, curr_target_date)

        # 8. ML Forecasting
        ml_models = MetalsForecastingModels()
        ml_cols = ["rsi_14", "dist_sma_200_pct", "volatility_20d", "gold_silver_ratio", "dxy_return_20d", "us10y_change_20d"]
        valid_ml_cols = [c for c in ml_cols if c in features_df.columns]
        model_predictions = ml_models.train_and_predict(features_df, valid_ml_cols, "future_gold_ret_5d", curr_target_date)

        # 9. Composite Scores & Probabilistic Signals
        scorer = CompositeScorer()
        scores = scorer.calculate_scores(feature_row, analogue_results.get("forward_statistics", {}))

        signals = ProbabilisticSignalEngine.generate_signal(
            instrument="GOLD",
            current_price=float(feature_row.get("gold_price", 0.0)),
            composite_score=scores["composite_score"],
            analogue_stats=analogue_results.get("forward_statistics", {}),
            model_predictions=model_predictions,
            market_regime=str(feature_row.get("market_regime", "Neutral")),
        )

        # 10. Forecast Performance Tracker
        evaluator = ForecastEvaluator()
        evaluator.log_prediction({
            "prediction_date": curr_target_date,
            "instrument": "GOLD",
            "horizon_signals": signals["horizon_signals"],
        })
        evaluator.evaluate_all_predictions(gold_df)

        # 11. Alerts
        alert_engine = AlertEngine()
        alerts = alert_engine.detect_alerts(curr_target_date, feature_row)

        # 12. Summaries & Reports
        gold_sum = {
            "price": float(feature_row.get("gold_price", 0.0)),
            "inr_price_10g": float(feature_row.get("inr_price_10g", 0.0)) if "inr_price_10g" in feature_row else None,
            "daily_move_pct": float(feature_row.get("gold_return_1d", 0.0) * 100),
        }
        silver_sum = {
            "price": float(feature_row.get("silver_price", 0.0)) if "silver_price" in feature_row else None,
            "inr_price_kg": float(feature_row.get("inr_price_kg", 0.0)) if "inr_price_kg" in feature_row else None,
            "daily_move_pct": float(feature_row.get("silver_return_1d", 0.0) * 100) if "silver_return_1d" in feature_row else 0.0,
        }
        ratio_sum = {
            "gold_silver_ratio": float(feature_row.get("gold_silver_ratio", 0.0)) if "gold_silver_ratio" in feature_row else None,
            "ratio_zscore": float(feature_row.get("gold_silver_ratio_zscore", 0.0)) if "gold_silver_ratio_zscore" in feature_row else None,
        }
        macro_sum = {
            "macro_regime": str(feature_row.get("macro_regime", "Neutral")),
            "dxy_close": float(feature_row.get("dxy_close", 0.0)) if "dxy_close" in feature_row else None,
            "dxy_return_20d": float(feature_row.get("dxy_return_20d", 0.0) * 100) if "dxy_return_20d" in feature_row else 0.0,
            "us10y_close": float(feature_row.get("us10y_close", 0.0)) if "us10y_close" in feature_row else None,
        }
        regime_sum = {
            "market_regime": str(feature_row.get("market_regime", "Neutral")),
        }
        quality_sum = {
            "valid_records": audit_reports.get("GOLD", {}).get("valid_records", 0),
            "suspicious_records": audit_reports.get("GOLD", {}).get("suspicious_records", 0),
            "warnings": audit_reports.get("GOLD", {}).get("warnings", []),
        }

        report_gen = ReportGenerator()
        report_file = report_gen.generate_daily_report(
            target_date=curr_target_date,
            gold_summary=gold_sum,
            silver_summary=silver_sum,
            ratio_summary=ratio_sum,
            macro_summary=macro_sum,
            regime_summary=regime_sum,
            analogue_summary=analogue_results,
            signals_summary={"gold": signals},
            scores_summary=scores,
            quality_summary=quality_sum,
        )

        logger.info(f"=== Pipeline execution complete. Generated report at: {report_file} ===")
        return {"status": "success", "date": curr_target_date, "report": report_file}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run daily precious metals research pipeline.")
    parser.add_argument("--date", type=str, help="Target date YYYY-MM-DD", default=None)
    args = parser.parse_args()

    runner = PipelineRunner()
    runner.run_daily_pipeline(target_date=args.date)
