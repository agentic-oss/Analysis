"""
End-to-End Orchestration Pipeline for Gold & Silver Intelligence.
Executes daily data collection, validation, feature store generation, multi-factor analysis,
historical analogue search, forecasting, scoring, backtesting, and daily report generation.
"""

from datetime import datetime
import json
import logging
import os
import sys
from typing import Dict, Any, Optional

import pandas as pd
import yaml

from src.ingestion.provider import YFinanceProvider, convert_usd_to_inr_gold_silver
from src.validation.validator import DataValidator
from src.storage.manager import StorageManager
from src.indicators.technical import calculate_technical_indicators, detect_technical_signals
from src.metals.gold import GoldAnalyzer
from src.ratios.silver_ratio import SilverRelativeValueAnalyzer
from src.regimes.classifier import RegimeClassifier
from src.patterns.analogue import AnalogueEngine
from src.features.builder import FeatureStoreBuilder
from src.forecasting.models import MetalsForecaster
from src.scoring.engine import ScoringAndAlertEngine
from src.backtesting.engine import Backtester
from src.reporting.generator import DailyReportGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PipelineRunner")


class PipelineRunner:
    """Orchestrates the entire end-to-end Gold & Silver market data and research pipeline."""

    def __init__(self, config_path: str = "config/config.yaml", instruments_path: str = "config/instruments.json"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        with open(instruments_path, "r") as f:
            self.instruments = json.load(f).get("instruments", [])

        self.provider = YFinanceProvider()
        self.validator = DataValidator()
        self.storage = StorageManager(base_dir=self.config["system"]["data_dir"])
        self.gold_analyzer = GoldAnalyzer()
        self.silver_analyzer = SilverRelativeValueAnalyzer()
        self.classifier = RegimeClassifier()
        self.analogue_engine = AnalogueEngine()
        self.feature_builder = FeatureStoreBuilder()
        self.scoring_engine = ScoringAndAlertEngine(alert_thresholds=self.config.get("alert_thresholds"))
        self.report_generator = DailyReportGenerator()

    def run_daily_pipeline(self, as_of_date: Optional[str] = None):
        as_of_date = as_of_date or datetime.now().strftime("%Y-%m-%d")
        logger.info(f"=== Starting Daily Pipeline Execution for Date: {as_of_date} ===")

        # 1. Fetch & Store Ingestion Datasets
        raw_datasets = {}
        for inst in self.instruments:
            if not inst.get("enabled", True):
                continue
            sym = inst["symbol"]
            ticker = inst.get("ticker", sym)
            df = self.provider.get_historical_prices(
                symbol=sym,
                ticker=ticker,
                start_date=(pd.Timestamp(as_of_date) - pd.Timedelta(days=1000)).strftime("%Y-%m-%d"),
                end_date=as_of_date
            )
            raw_datasets[sym] = df
            self.storage.save_raw_snapshot(sym, df, as_of_date)

        # 2. Data Validation & Quality Reporting
        quality_report = self.validator.validate_and_save_batch(raw_datasets, as_of_date)

        # 3. Calculate Technical Indicators & Master Store
        processed_metals = {}
        processed_macro = {}

        for sym, df in raw_datasets.items():
            if df.empty:
                continue
            df_ind = calculate_technical_indicators(df)
            cat = "metals" if "GOLD" in sym or "SILVER" in sym else "macro"
            self.storage.save_processed_dataset(cat, sym, df_ind)
            if cat == "metals":
                processed_metals[sym] = df_ind
            else:
                processed_macro[sym] = df_ind

        gold_df = processed_metals.get("GOLD", pd.DataFrame())
        silver_df = processed_metals.get("SILVER", pd.DataFrame())

        # 4. Relative Value & Gold/Silver Ratio Analysis
        ratio_df = pd.DataFrame()
        ratio_extremes = {}
        if not gold_df.empty and not silver_df.empty:
            ratio_df = self.silver_analyzer.calculate_ratio_metrics(gold_df, silver_df)
            ratio_extremes = self.silver_analyzer.analyze_ratio_extremes(ratio_df)

        # 5. Build Feature Dataset
        events_df = self.provider.get_economic_indicators()
        gold_features = self.feature_builder.build_daily_features(
            metal_df=gold_df,
            ratio_df=ratio_df,
            macro_dfs=processed_macro,
            events_df=events_df,
            symbol="GOLD"
        )
        silver_features = self.feature_builder.build_daily_features(
            metal_df=silver_df,
            ratio_df=ratio_df,
            macro_dfs=processed_macro,
            events_df=events_df,
            symbol="SILVER"
        )

        if not gold_features.empty:
            self.storage.save_features(gold_features, as_of_date)

        # 6. Regime Classification
        us10y_df = processed_macro.get("US10Y", pd.DataFrame())
        dxy_df = processed_macro.get("DXY", pd.DataFrame())
        vix_df = processed_macro.get("VIX", pd.DataFrame())
        oil_df = processed_macro.get("OIL_BRENT", pd.DataFrame())
        sp_df = processed_macro.get("SP500", pd.DataFrame())

        rates_chg = us10y_df["close"].diff(20).iloc[-1] if not us10y_df.empty else 0.0
        dxy_ret = dxy_df["close"].pct_change(20).iloc[-1] * 100.0 if not dxy_df.empty else 0.0
        vix_val = vix_df["close"].iloc[-1] if not vix_df.empty else 18.0
        oil_ret = oil_df["close"].pct_change(20).iloc[-1] * 100.0 if not oil_df.empty else 0.0
        sp_ret = sp_df["close"].pct_change(20).iloc[-1] * 100.0 if not sp_df.empty else 0.0

        macro_regime_res = self.classifier.classify_macro_regime(
            rates_change_20d=float(rates_chg if pd.notna(rates_chg) else 0.0),
            dxy_return_20d=float(dxy_ret if pd.notna(dxy_ret) else 0.0),
            vix_level=float(vix_val if pd.notna(vix_val) else 18.0),
            oil_return_20d=float(oil_ret if pd.notna(oil_ret) else 0.0),
            equity_return_20d=float(sp_ret if pd.notna(sp_ret) else 0.0)
        )
        gold_market_regime = self.classifier.classify_market_regime(gold_df)
        silver_market_regime = self.classifier.classify_market_regime(silver_df)

        # 7. Historical Analogue Engine
        gold_analogues = self.analogue_engine.find_analogues(gold_features, target_instrument="GOLD")
        silver_analogues = self.analogue_engine.find_analogues(silver_features, target_instrument="SILVER")

        # 8. Forecasting Models
        g_forecaster = MetalsForecaster(target_horizon=5)
        g_forecast_5d = g_forecaster.train_and_predict(gold_features)
        s_forecaster = MetalsForecaster(target_horizon=5)
        s_forecast_5d = s_forecaster.train_and_predict(silver_features)

        # 9. Scoring & Alerts
        g_tech_signals = detect_technical_signals(gold_df)
        s_tech_signals = detect_technical_signals(silver_df)

        g_rsi = gold_df["rsi_14"].iloc[-1] if "rsi_14" in gold_df.columns else 50.0
        s_rsi = silver_df["rsi_14"].iloc[-1] if "rsi_14" in silver_df.columns else 50.0

        gold_scores = self.scoring_engine.compute_scores(
            tech_signals=g_tech_signals,
            rsi=g_rsi,
            macro_regime=macro_regime_res["macro_regime"],
            analogue_win_rate=gold_analogues.get("forward_return_statistics", {}).get("5d", {}).get("positive_prob_pct", 50.0),
            model_prob_pos=g_forecast_5d.get("probability_positive", 50.0)
        )

        silver_scores = self.scoring_engine.compute_scores(
            tech_signals=s_tech_signals,
            rsi=s_rsi,
            gs_ratio_zscore=ratio_extremes.get("zscore", 0.0),
            macro_regime=macro_regime_res["macro_regime"],
            analogue_win_rate=silver_analogues.get("forward_return_statistics", {}).get("5d", {}).get("positive_prob_pct", 50.0),
            model_prob_pos=s_forecast_5d.get("probability_positive", 50.0)
        )

        alerts = self.scoring_engine.detect_alerts(
            symbol="GOLD",
            price_change_pct=gold_df["close"].pct_change().iloc[-1] * 100.0 if not gold_df.empty else 0.0,
            rsi=g_rsi,
            vol_zscore=0.0,
            gs_ratio_zscore=ratio_extremes.get("zscore", 0.0),
            tech_signals=g_tech_signals
        )

        # 10. Assemble Analysis Structure
        analysis_payload = {
            "date": as_of_date,
            "gold": {
                "price": float(gold_df["close"].iloc[-1]) if not gold_df.empty else 0.0,
                "return_1d": float(gold_df["close"].pct_change().iloc[-1] * 100.0) if not gold_df.empty else 0.0,
                "return_5d": float(gold_df["close"].pct_change(5).iloc[-1] * 100.0) if not gold_df.empty else 0.0,
                "return_20d": float(gold_df["close"].pct_change(20).iloc[-1] * 100.0) if not gold_df.empty else 0.0,
                "trend_regime": gold_market_regime.get("trend_regime"),
                "volatility_regime": gold_market_regime.get("volatility_regime"),
                "rsi_14": float(g_rsi),
                "macd": float(gold_df["macd"].iloc[-1]) if "macd" in gold_df.columns else 0.0,
                "atr_14": float(gold_df["atr_14"].iloc[-1]) if "atr_14" in gold_df.columns else 0.0,
                "dist_sma_200_pct": float(gold_df["dist_sma_200_pct"].iloc[-1]) if "dist_sma_200_pct" in gold_df.columns else 0.0,
                "dist_52w_high_pct": float(gold_df["dist_52w_high_pct"].iloc[-1]) if "dist_52w_high_pct" in gold_df.columns else 0.0,
                "composite_score": gold_scores["composite_score"],
                "research_bias": g_forecast_5d.get("direction", "NEUTRAL"),
                "top_analogues": gold_analogues.get("top_analogues", []),
                "forward_statistics": {"5d": g_forecast_5d}
            },
            "silver": {
                "price": float(silver_df["close"].iloc[-1]) if not silver_df.empty else 0.0,
                "return_1d": float(silver_df["close"].pct_change().iloc[-1] * 100.0) if not silver_df.empty else 0.0,
                "return_5d": float(silver_df["close"].pct_change(5).iloc[-1] * 100.0) if not silver_df.empty else 0.0,
                "return_20d": float(silver_df["close"].pct_change(20).iloc[-1] * 100.0) if not silver_df.empty else 0.0,
                "trend_regime": silver_market_regime.get("trend_regime"),
                "volatility_regime": silver_market_regime.get("volatility_regime"),
                "rsi_14": float(s_rsi),
                "atr_14": float(silver_df["atr_14"].iloc[-1]) if "atr_14" in silver_df.columns else 0.0,
                "composite_score": silver_scores["composite_score"],
                "research_bias": s_forecast_5d.get("direction", "NEUTRAL"),
                "top_analogues": silver_analogues.get("top_analogues", []),
                "forward_statistics": {"5d": s_forecast_5d}
            },
            "gold_silver_ratio": ratio_extremes,
            "macro": {
                "macro_regime": macro_regime_res["macro_regime"],
                "dxy_close": float(dxy_df["close"].iloc[-1]) if not dxy_df.empty else 0.0,
                "us10y_close": float(us10y_df["close"].iloc[-1]) if not us10y_df.empty else 0.0,
                "oil_brent_close": float(oil_df["close"].iloc[-1]) if not oil_df.empty else 0.0,
                "vix_close": float(vix_val),
                "nifty_close": float(processed_macro.get("NIFTY50", pd.DataFrame())["close"].iloc[-1]) if "NIFTY50" in processed_macro and not processed_macro["NIFTY50"].empty else 0.0,
                "usdinr_close": float(processed_macro.get("USDINR", pd.DataFrame())["close"].iloc[-1]) if "USDINR" in processed_macro and not processed_macro["USDINR"].empty else 0.0
            },
            "data_quality": quality_report.to_dict(),
            "alerts": alerts
        }

        # 11. Save Reports & Analysis JSON
        self.storage.save_analysis_json(analysis_payload, as_of_date)
        if alerts:
            self.storage.save_alerts_json({"date": as_of_date, "alerts": alerts}, as_of_date)

        md_report_path = os.path.join(self.config["system"]["reports_dir"], "daily", f"{as_of_date}.md")
        self.report_generator.generate_markdown_report(analysis_payload, md_report_path)

        json_report_path = os.path.join(self.config["system"]["data_dir"], "analysis", "daily", f"{as_of_date}.json")
        self.report_generator.generate_json_report(analysis_payload, json_report_path)

        logger.info(f"=== Pipeline completed successfully for {as_of_date} ===")
        return analysis_payload


if __name__ == "__main__":
    runner = PipelineRunner()
    date_arg = sys.argv[1] if len(sys.argv) > 1 else None
    runner.run_daily_pipeline(date_arg)
