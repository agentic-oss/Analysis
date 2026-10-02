import os
import sys
import json
import logging
import yaml
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# Imports from src packages
from src.ingestion.provider import YFinanceDataProvider
from src.ingestion.converter import UnitCurrencyConverter
from src.validation.validator import DataValidator
from src.storage.manager import StorageManager
from src.indicators.technical import TechnicalIndicators
from src.metals.gold_analysis import GoldMultiFactorAnalyzer
from src.ratios.ratio_analysis import GoldSilverRatioAnalyzer
from src.regimes.macro_regime import MacroRegimeClassifier
from src.regimes.market_regime import MarketRegimeClassifier
from src.patterns.analogue_engine import HistoricalAnalogueEngine
from src.features.feature_builder import DailyFeatureBuilder
from src.scoring.score_engine import ScoreEngine
from src.scoring.alert_engine import AlertEngine
from src.forecasting.probability_engine import ForwardProbabilityEngine
from src.backtesting.engine import StrategyBacktester
from src.evaluation.forecast_evaluator import ForecastEvaluator
from src.reporting.report_generator import ReportGenerator

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PipelineRunner")

def create_synthetic_historical_data(date_str: str) -> dict:
    """Generates deterministic baseline history if live fetch is unavailable."""
    dates = pd.date_range(end=date_str, periods=300, freq="B")
    dt_strs = [d.strftime("%Y-%m-%d") for d in dates]
    np.random.seed(42)

    # Gold
    g_close = 2000.0 + np.cumsum(np.random.normal(0.5, 12.0, 300))
    gold_df = pd.DataFrame({
        "date": dt_strs, "symbol": "GOLD", "close": g_close,
        "open": g_close * 0.998, "high": g_close * 1.008, "low": g_close * 0.992, "volume": 50000
    })

    # Silver
    s_close = 23.0 + np.cumsum(np.random.normal(0.01, 0.35, 300))
    silver_df = pd.DataFrame({
        "date": dt_strs, "symbol": "SILVER", "close": s_close,
        "open": s_close * 0.995, "high": s_close * 1.010, "low": s_close * 0.988, "volume": 20000
    })

    # USDINR
    inr_close = 82.0 + np.cumsum(np.random.normal(0.01, 0.15, 300))
    usdinr_df = pd.DataFrame({"date": dt_strs, "symbol": "USDINR", "close": inr_close})

    # DXY
    dxy_close = 102.0 + np.cumsum(np.random.normal(0.02, 0.3, 300))
    dxy_df = pd.DataFrame({"date": dt_strs, "symbol": "DXY", "close": dxy_close})

    # US10Y
    us10y_close = 4.0 + np.cumsum(np.random.normal(0.0, 0.03, 300))
    us10y_df = pd.DataFrame({"date": dt_strs, "symbol": "US10Y", "close": us10y_close})

    # VIX
    vix_close = np.clip(16.0 + np.cumsum(np.random.normal(0.0, 0.8, 300)), 10, 40)
    vix_df = pd.DataFrame({"date": dt_strs, "symbol": "VIX", "close": vix_close})

    # SP500
    sp500_close = 5000.0 + np.cumsum(np.random.normal(2.0, 25.0, 300))
    sp500_df = pd.DataFrame({"date": dt_strs, "symbol": "SP500", "close": sp500_close})

    # Crude Oil WTI
    wti_close = 75.0 + np.cumsum(np.random.normal(0.05, 1.2, 300))
    wti_df = pd.DataFrame({"date": dt_strs, "symbol": "WTI", "close": wti_close})

    return {
        "GOLD": gold_df, "SILVER": silver_df, "USDINR": usdinr_df,
        "DXY": dxy_df, "US10Y": us10y_df, "VIX": vix_df,
        "SP500": sp500_df, "WTI": wti_df
    }

def run_pipeline(target_date: str = None):
    if not target_date:
        target_date = datetime.now().strftime("%Y-%m-%d")

    logger.info(f"Starting Precious Metals Intelligence Pipeline for date: {target_date}")

    # Load configs
    with open("config/instruments.json") as f:
        instruments_cfg = json.load(f)
    with open("config/config.yaml") as f:
        config_yaml = yaml.safe_load(f)

    storage = StorageManager(config_yaml["system"]["data_dir"])
    provider = YFinanceDataProvider()
    converter = UnitCurrencyConverter()
    validator = DataValidator()

    # 1. Ingest / Fetch Data
    raw_data_dict = {}
    for inst in instruments_cfg.get("instruments", []):
        sym = inst["symbol"]
        df = provider.fetch_symbol_data(inst, start_date="2022-01-01")
        if df is not None and not df.empty:
            raw_data_dict[sym] = df
            storage.save_raw_data(df, target_date, sym)

    if "GOLD" not in raw_data_dict or len(raw_data_dict["GOLD"]) < 20:
        logger.warning("Live data fetch returned empty/incomplete dataset. Utilizing baseline synthetic dataset.")
        raw_data_dict = create_synthetic_historical_data(target_date)

    # 2. Validate Data
    all_raw = pd.concat(list(raw_data_dict.values()), ignore_index=True)
    val_df, quality_report = validator.validate_dataframe(all_raw)
    storage.save_quality_json(quality_report, target_date)

    # 3. Process Technical Indicators
    gold_df = TechnicalIndicators.calculate_all(raw_data_dict["GOLD"])
    silver_df = TechnicalIndicators.calculate_all(raw_data_dict["SILVER"])

    storage.save_processed_data(gold_df, "metals", "GOLD")
    storage.save_processed_data(silver_df, "metals", "SILVER")

    # 4. Macro & Correlations
    macro_list = []
    for k in ["USDINR", "DXY", "US10Y", "VIX", "SP500", "WTI"]:
        if k in raw_data_dict:
            macro_list.append(raw_data_dict[k])
    macro_df = pd.concat(macro_list, ignore_index=True) if macro_list else pd.DataFrame()

    gold_analyzer = GoldMultiFactorAnalyzer()
    corr_results = gold_analyzer.analyze_cross_market_correlations(gold_df, macro_df)

    # 5. Relative Value Ratio
    usdinr_s = raw_data_dict.get("USDINR", pd.DataFrame()).set_index("date")["close"] if "USDINR" in raw_data_dict else pd.Series()
    gold_df = converter.process_metal_dataframe(gold_df, usdinr_s)
    silver_df = converter.process_metal_dataframe(silver_df, usdinr_s)

    ratio_analyzer = GoldSilverRatioAnalyzer()
    ratio_df = ratio_analyzer.analyze_ratio(
        gold_df.set_index("date")["close"],
        silver_df.set_index("date")["close"]
    )
    ratio_summary = ratio_analyzer.get_latest_summary(ratio_df)

    # 6. Regimes
    piv = macro_df.pivot(index="date", columns="symbol", values="close") if not macro_df.empty else pd.DataFrame()
    dxy_ret_20 = float(piv["DXY"].pct_change(20).iloc[-1]) if "DXY" in piv.columns and len(piv["DXY"].dropna()) >= 20 else 0.0
    us10y_change_20 = float(piv["US10Y"].diff(20).iloc[-1]) if "US10Y" in piv.columns and len(piv["US10Y"].dropna()) >= 20 else 0.0
    vix_val = float(piv["VIX"].iloc[-1]) if "VIX" in piv.columns and not piv["VIX"].empty else 18.0
    sp500_ret_20 = float(piv["SP500"].pct_change(20).iloc[-1]) if "SP500" in piv.columns and len(piv["SP500"].dropna()) >= 20 else 0.0
    oil_col = "WTI" if "WTI" in piv.columns else ("BRENT" if "BRENT" in piv.columns else None)
    oil_ret_20 = float(piv[oil_col].pct_change(20).iloc[-1]) if oil_col and len(piv[oil_col].dropna()) >= 20 else 0.0

    macro_latest_snap = {
        "dxy_return_20d": dxy_ret_20,
        "us10y_change_20d": us10y_change_20,
        "vix_level": vix_val,
        "sp500_return_20d": sp500_ret_20,
        "oil_return_20d": oil_ret_20
    }
    macro_regime = MacroRegimeClassifier().classify_regime(macro_latest_snap)

    mkt_classifier = MarketRegimeClassifier()
    gold_mkt_regime = mkt_classifier.classify_market_regime(gold_df)
    silver_mkt_regime = mkt_classifier.classify_market_regime(silver_df)

    # 7. Features & Historical Analogues
    feature_builder = DailyFeatureBuilder()
    gold_feats = feature_builder.build_feature_dataset(gold_df, macro_df, ratio_df, macro_regime, gold_mkt_regime)
    silver_feats = feature_builder.build_feature_dataset(silver_df, macro_df, ratio_df, macro_regime, silver_mkt_regime)

    storage.save_features(gold_feats, target_date)

    curr_gold_feats = gold_feats.iloc[-1]
    hist_gold_feats = gold_feats.iloc[:-1]

    analogue_engine = HistoricalAnalogueEngine(top_k=15)
    gold_analogues = analogue_engine.find_analogues(curr_gold_feats, hist_gold_feats)

    # 8. Scores & Alerts
    score_engine = ScoreEngine()
    gold_scores = score_engine.compute_scores(
        gold_df.iloc[-1], macro_regime, ratio_summary, gold_analogues
    )
    silver_scores = score_engine.compute_scores(
        silver_df.iloc[-1], macro_regime, ratio_summary, gold_analogues
    )

    alert_engine = AlertEngine()
    alerts_dict = alert_engine.detect_alerts(target_date, gold_df, silver_df, ratio_summary, macro_regime)
    storage.save_alerts_json(alerts_dict, target_date)

    # 9. Forecasting Probabilities
    prob_engine = ForwardProbabilityEngine()
    gold_forecasts = prob_engine.generate_horizon_forecasts(
        "GOLD", gold_feats, gold_scores["composite_score"], gold_mkt_regime["volatility_regime"]
    )
    silver_forecasts = prob_engine.generate_horizon_forecasts(
        "SILVER", silver_feats, silver_scores["composite_score"], silver_mkt_regime["volatility_regime"]
    )

    predictions_payload = {
        "date": target_date,
        "gold_forecasts": gold_forecasts,
        "silver_forecasts": silver_forecasts
    }
    storage.save_predictions_json(predictions_payload, target_date)

    # 10. Daily Analysis JSON
    daily_analysis_json = {
        "date": target_date,
        "gold": {
            "price": float(gold_df.iloc[-1]["close"]),
            "return_1d_pct": float(gold_df.iloc[-1].get("return_1d", 0.0) * 100),
            "return_5d_pct": float(gold_df.iloc[-1].get("return_5d", 0.0) * 100),
            "return_20d_pct": float(gold_df.iloc[-1].get("return_20d", 0.0) * 100),
            "technical_score": gold_scores["technical_score"],
            "macro_score": gold_scores["macro_score"],
            "regime": gold_mkt_regime["composite_label"],
            "research_bias": gold_scores["score_interpretation"],
            "rsi_14": float(gold_df.iloc[-1].get("rsi_14", 50.0)),
            "macd": float(gold_df.iloc[-1].get("macd", 0.0)),
            "dist_200dma": float(gold_df.iloc[-1].get("dist_sma_200", 0.0)),
            "scores": gold_scores,
            "forward_statistics": gold_forecasts
        },
        "silver": {
            "price": float(silver_df.iloc[-1]["close"]),
            "return_1d_pct": float(silver_df.iloc[-1].get("return_1d", 0.0) * 100),
            "return_5d_pct": float(silver_df.iloc[-1].get("return_5d", 0.0) * 100),
            "return_20d_pct": float(silver_df.iloc[-1].get("return_20d", 0.0) * 100),
            "technical_score": silver_scores["technical_score"],
            "macro_score": silver_scores["macro_score"],
            "regime": silver_mkt_regime["composite_label"],
            "research_bias": silver_scores["score_interpretation"],
            "rsi_14": float(silver_df.iloc[-1].get("rsi_14", 50.0)),
            "macd": float(silver_df.iloc[-1].get("macd", 0.0)),
            "dist_200dma": float(silver_df.iloc[-1].get("dist_sma_200", 0.0)),
            "scores": silver_scores,
            "forward_statistics": silver_forecasts
        },
        "gold_silver_ratio": ratio_summary,
        "macro": macro_regime,
        "correlations": corr_results,
        "analogues": gold_analogues,
        "alerts": alerts_dict,
        "data_quality": quality_report
    }

    storage.save_analysis_json(daily_analysis_json, target_date)

    # 11. Generate Markdown Research Report
    report_gen = ReportGenerator()
    report_path = report_gen.generate_markdown_report(target_date, daily_analysis_json)
    logger.info(f"Daily Markdown research report saved to: {report_path}")

    # 12. Run Backtesting & Forecast Performance Tracking
    backtester = StrategyBacktester()
    signals_series = pd.Series(np.where(gold_feats["rsi_14"] < 40, 1, np.where(gold_feats["rsi_14"] > 60, -1, 0)), index=gold_feats.index)
    bt_res = backtester.run_backtest(gold_df, signals_series)

    evaluator = ForecastEvaluator()
    sample_preds = [{
        "date": target_date, "symbol": "GOLD", "horizon_days": 10,
        "positive_return_probability": gold_forecasts["forecasts"]["10d"]["positive_return_probability"],
        "expected_return_pct": gold_forecasts["forecasts"]["10d"]["expected_return_pct"]
    }]
    perf_metrics = evaluator.evaluate_historical_forecasts(sample_preds, gold_df)

    with open("data/models/model-performance.json", "w") as f:
        json.dump({"date": target_date, "backtest": bt_res, "performance": perf_metrics}, f, indent=2)

    logger.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    date_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(date_arg)
