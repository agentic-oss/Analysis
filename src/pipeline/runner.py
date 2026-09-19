import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta, timezone
import yaml
import pandas as pd
import numpy as np

from src.ingestion.provider import YFinanceProvider, MockMarketDataProvider
from src.validation.validator import DataValidator
from src.storage.storage_manager import StorageManager
from src.indicators.technical import TechnicalIndicators
from src.macro.macro_analyzer import MacroAnalyzer
from src.metals.metal_analyzer import MetalAnalyzer
from src.ratios.ratio_analyzer import RatioAnalyzer
from src.regimes.regime_detector import RegimeDetector
from src.patterns.analogue_engine import HistoricalAnalogueEngine
from src.features.feature_builder import FeatureBuilder
from src.scoring.scoring_engine import ScoringEngine
from src.forecasting.probabilistic_engine import ProbabilisticForecaster
from src.models.baseline_models import BaselineModels
from src.evaluation.evaluator import ForecastEvaluator
from src.backtesting.backtest_engine import BacktestEngine
from src.reporting.report_generator import ReportGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pipeline")

class DailyPipelineRunner:
    """Orchestrates the end-to-end Gold & Silver analysis pipeline."""

    def __init__(self, config_path: str = "config/config.yaml", instruments_path: str = "config/instruments.json"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        with open(instruments_path, "r") as f:
            self.instruments_data = json.load(f)
        self.storage = StorageManager(self.config['system'].get('data_dir', 'data'))

    def run_pipeline(self, target_date: str, use_mock: bool = False):
        logger.info(f"--- Starting Daily Pipeline Run for {target_date} ---")

        # 1. Fetch historical / daily market data
        end_dt = datetime.strptime(target_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=365 * self.config['system'].get('default_lookback_years', 5))
        start_str = start_dt.strftime("%Y-%m-%d")

        if use_mock:
            # Generate deterministic synthetic data for pipeline verification
            logger.info("Using mock data provider.")
            dates = pd.date_range(start_str, target_date, freq="B").strftime("%Y-%m-%d")
            np.random.seed(42)
            gold_p = 2000 + np.cumsum(np.random.randn(len(dates)) * 12)
            silver_p = 24 + np.cumsum(np.random.randn(len(dates)) * 0.3)

            gold_df = pd.DataFrame({"date": dates, "open": gold_p, "high": gold_p + 10, "low": gold_p - 10, "close": gold_p, "volume": 10000})
            silver_df = pd.DataFrame({"date": dates, "open": silver_p, "high": silver_p + 0.5, "low": silver_p - 0.5, "close": silver_p, "volume": 5000})
            dxy_df = pd.DataFrame({"date": dates, "close": 103 + np.cumsum(np.random.randn(len(dates)) * 0.2)})

            provider = MockMarketDataProvider({
                "GOLD_USD": gold_df,
                "SILVER_USD": silver_df,
                "DXY": dxy_df
            })
        else:
            provider = YFinanceProvider(
                max_retries=self.config['providers']['max_retries'],
                backoff_factor=self.config['providers']['backoff_factor'],
                timeout=self.config['providers']['timeout_seconds']
            )

        # Download & validate data for enabled instruments
        instrument_dfs = {}
        data_quality_reports = {}

        for inst in self.instruments_data.get("instruments", []):
            if not inst.get("enabled", True):
                continue
            sym = inst["symbol"]
            ticker = inst["ticker"]

            df = provider.fetch_historical_prices(sym, ticker, start_str, target_date)
            if df.empty and use_mock:
                # Fallback for mock missing keys
                dates = pd.date_range(start_str, target_date, freq="B").strftime("%Y-%m-%d")
                df = pd.DataFrame({"date": dates, "open": 100.0, "high": 105.0, "low": 95.0, "close": 100.0, "volume": 1000})

            val_df, q_summary = DataValidator.validate_df(df, sym)
            instrument_dfs[sym] = val_df
            data_quality_reports[sym] = q_summary

            # Save processed instrument price time-series
            self.storage.save_df(val_df, f"processed/metals/{sym}")

        # Save data quality log
        quality_log = {
            "date": target_date,
            "reports": data_quality_reports
        }
        self.storage.save_json(quality_log, f"quality/{target_date}")

        # 2. Compute Technical Indicators
        gold_df = instrument_dfs.get("GOLD_USD", pd.DataFrame())
        silver_df = instrument_dfs.get("SILVER_USD", pd.DataFrame())
        dxy_df = instrument_dfs.get("DXY", pd.DataFrame())

        if not gold_df.empty:
            gold_df = TechnicalIndicators.calculate_all(gold_df, self.config)
            self.storage.save_df(gold_df, "processed/metals/GOLD_USD")

        if not silver_df.empty:
            silver_df = TechnicalIndicators.calculate_all(silver_df, self.config)
            self.storage.save_df(silver_df, "processed/metals/SILVER_USD")

        # 3. Ratio & Cross-market Analysis
        ratio_res = RatioAnalyzer.calculate_gold_silver_ratio(gold_df, silver_df)

        gold_analysis = MetalAnalyzer.analyze_metal(gold_df, "GOLD_USD", dxy_df)
        silver_analysis = MetalAnalyzer.analyze_metal(silver_df, "SILVER_USD", dxy_df)

        macro_data_summary = {
            "dxy_return_20d": float(dxy_df['close'].pct_change(20).iloc[-1] * 100) if not dxy_df.empty and len(dxy_df) > 20 else 0.0,
            "us10y_yield": 3.8,
            "us10y_change_20d": -0.1,
            "vix": 18.5,
            "sp500_return_20d": 1.2,
            "oil_return_20d": -0.5
        }
        macro_regime = RegimeDetector.detect_macro_regime(macro_data_summary)
        gold_market_regime = RegimeDetector.detect_market_regime(gold_df)
        silver_market_regime = RegimeDetector.detect_market_regime(silver_df)

        # 4. Feature Store Building
        gold_feat = FeatureBuilder.build_feature_dataset(gold_df, horizons=self.config['forecasting']['horizons'])
        silver_feat = FeatureBuilder.build_feature_dataset(silver_df, horizons=self.config['forecasting']['horizons'])

        self.storage.save_df(gold_feat, "features/daily/GOLD_USD_features")
        self.storage.save_df(silver_feat, "features/daily/SILVER_USD_features")

        # 5. Historical Analogue Search
        gold_current_idx = len(gold_feat) - 1
        silver_current_idx = len(silver_feat) - 1

        feature_cols = ['rsi_14', 'macd', 'atr_14', 'volatility_20d', 'dist_sma_200_pct']

        gold_analogues = HistoricalAnalogueEngine.find_analogues(
            gold_feat, feature_cols, gold_current_idx,
            top_k=self.config['forecasting']['analogue_top_k'],
            horizons=self.config['forecasting']['horizons']
        )
        silver_analogues = HistoricalAnalogueEngine.find_analogues(
            silver_feat, feature_cols, silver_current_idx,
            top_k=self.config['forecasting']['analogue_top_k'],
            horizons=self.config['forecasting']['horizons']
        )

        # 6. Scoring & Probabilistic Forecasts
        gold_scores = ScoringEngine.calculate_scores(gold_analysis, macro_regime, ratio_res, gold_analogues, self.config)
        silver_scores = ScoringEngine.calculate_scores(silver_analysis, macro_regime, ratio_res, silver_analogues, self.config)

        gold_forecasts = ProbabilisticForecaster.generate_forecasts(gold_analogues, gold_scores, self.config['forecasting']['horizons'])
        silver_forecasts = ProbabilisticForecaster.generate_forecasts(silver_analogues, silver_scores, self.config['forecasting']['horizons'])

        # 7. ML Baseline Models & Evaluation
        ml_res = BaselineModels.train_and_predict(gold_feat, feature_cols, horizon=5)
        self.storage.save_json(ml_res, "models/model-performance")

        # 8. Strategy Backtesting
        if not gold_df.empty and 'rsi_14' in gold_df.columns:
            signals = (gold_df['rsi_14'] < 40).astype(int) - (gold_df['rsi_14'] > 60).astype(int)
            bt_results = BacktestEngine.run_signal_backtest(
                gold_df, signals,
                initial_capital=self.config['backtest']['initial_capital'],
                transaction_cost_bps=self.config['backtest']['transaction_cost_bps'],
                slippage_bps=self.config['backtest']['slippage_bps']
            )
            self.storage.save_json(bt_results, "backtests/gold_rsi_strategy")

        # 9. Reports & Alerts
        daily_json = {
            "date": target_date,
            "gold": {
                "price": gold_analysis.get("current_price", 0.0),
                "technical_score": gold_scores.get("technical_score", 0),
                "macro_score": gold_scores.get("macro_score", 0),
                "composite_score": gold_scores.get("composite_score", 0),
                "regime": gold_market_regime,
                "forecasts": gold_forecasts
            },
            "silver": {
                "price": silver_analysis.get("current_price", 0.0),
                "technical_score": silver_scores.get("technical_score", 0),
                "macro_score": silver_scores.get("macro_score", 0),
                "composite_score": silver_scores.get("composite_score", 0),
                "regime": silver_market_regime,
                "forecasts": silver_forecasts
            },
            "gold_silver_ratio": {
                "current": ratio_res.get("current_ratio", 0.0),
                "z_score": ratio_res.get("z_score", 0.0),
                "percentile": ratio_res.get("percentile", 50.0)
            },
            "macro": macro_regime,
            "data_quality": data_quality_reports.get("GOLD_USD", {})
        }

        self.storage.save_json(daily_json, f"analysis/daily/{target_date}")

        # Alerts
        alerts = []
        if ratio_res.get("is_extreme_high"):
            alerts.append({"type": "GOLD_SILVER_RATIO_EXTREME_HIGH", "value": ratio_res.get("current_ratio")})
        if gold_analysis.get("rsi_14", 50) > 70:
            alerts.append({"type": "GOLD_RSI_OVERBOUGHT", "value": gold_analysis.get("rsi_14")})
        elif gold_analysis.get("rsi_14", 50) < 30:
            alerts.append({"type": "GOLD_RSI_OVERSOLD", "value": gold_analysis.get("rsi_14")})

        self.storage.save_json({"date": target_date, "alerts": alerts}, f"analysis/alerts/{target_date}")

        # Daily Markdown Report
        markdown_rep = ReportGenerator.generate_daily_markdown_report(
            target_date, gold_analysis, silver_analysis, ratio_res,
            macro_regime, gold_forecasts, silver_forecasts,
            gold_analogues, silver_analogues, data_quality_reports.get("GOLD_USD", {})
        )

        rep_path = os.path.join(self.config['system'].get('reports_dir', 'reports'), f"daily/{target_date}.md")
        os.makedirs(os.path.dirname(rep_path), exist_ok=True)
        with open(rep_path, 'w', encoding='utf-8') as f:
            f.write(markdown_rep)

        logger.info(f"--- Successfully finished Daily Pipeline Run for {target_date} ---")
        return daily_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gold & Silver Daily Research Pipeline Runner")
    parser.add_argument("date", type=str, nargs="?", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"), help="Target trading date YYYY-MM-DD")
    parser.add_argument("--mock", action="store_true", help="Use mock data provider for testing/offline execution")
    args = parser.parse_args()

    runner = DailyPipelineRunner()
    runner.run_pipeline(args.date, use_mock=args.mock)
