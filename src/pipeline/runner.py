import os
import sys
import json
import yaml
import logging
import argparse
from datetime import datetime, timedelta
import pandas as pd

from src.ingestion.provider import YahooMarketDataProvider, SyntheticMarketDataProvider, convert_usd_to_inr_metrics
from src.validation.validator import DataValidator
from src.storage.manager import StorageManager
from src.indicators.technical import compute_all_technical_indicators
from src.metals.correlation import MultiAssetCorrelationAnalyzer
from src.ratios.gold_silver import GoldSilverRatioAnalyzer
from src.regimes.classifier import MacroRegimeClassifier, MarketRegimeClassifier
from src.patterns.analogue import HistoricalAnalogueEngine
from src.features.builder import DailyFeatureStoreBuilder
from src.models.forecaster import TimeSeriesModelEvaluator
from src.forecasting.confidence import PredictionConfidenceFramework
from src.scoring.composite import ScoringSystem
from src.scoring.alerts import AlertDetector
from src.evaluation.tracker import ForecastPerformanceTracker
from src.reporting.generator import ReportGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PipelineRunner")

def load_config():
    config_path = "config/config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}

def load_instruments():
    inst_path = "config/instruments.json"
    if os.path.exists(inst_path):
        with open(inst_path, "r") as f:
            return json.load(f).get("instruments", [])
    return []

def get_data_provider(config: dict):
    provider_type = config.get("system", {}).get("default_provider", "yahoo")
    if provider_type == "yahoo":
        return YahooMarketDataProvider()
    else:
        return SyntheticMarketDataProvider(seed=42)

def run_pipeline(target_date: str):
    logger.info(f"Starting daily precious metals pipeline execution for date {target_date}")

    config = load_config()
    instruments = load_instruments()

    storage = StorageManager()
    validator = DataValidator()

    # Attempt primary provider with fallback to synthetic provider
    try:
        provider = get_data_provider(config)
        logger.info(f"Initialized data provider: {provider.__class__.__name__}")
    except Exception as e:
        logger.warning(f"Failed to initialize configured provider, falling back to SyntheticMarketDataProvider: {e}")
        provider = SyntheticMarketDataProvider(seed=42)

    # 1. Collect market data for metals & macro instruments
    start_date = (pd.to_datetime(target_date) - timedelta(days=730)).strftime("%Y-%m-%d")

    def fetch_with_fallback(symbol: str) -> pd.DataFrame:
        try:
            df = provider.fetch_historical_series(symbol, start_date, target_date)
            if df.empty:
                logger.warning(f"Primary provider returned empty df for {symbol}, trying synthetic provider fallback.")
                df = SyntheticMarketDataProvider(seed=42).fetch_historical_series(symbol, start_date, target_date)
            return df
        except Exception as e:
            logger.warning(f"Error fetching {symbol} from primary provider ({e}), using synthetic fallback.")
            return SyntheticMarketDataProvider(seed=42).fetch_historical_series(symbol, start_date, target_date)

    gold_df = fetch_with_fallback("GC=F")
    silver_df = fetch_with_fallback("SI=F")
    dxy_df = fetch_with_fallback("DX-Y.NYB")
    us10y_df = fetch_with_fallback("^TNX")
    vix_df = fetch_with_fallback("^VIX")
    oil_df = fetch_with_fallback("CL=F")
    usdinr_df = fetch_with_fallback("INR=X")

    # 2. Validate data across all series and write aggregated quality report
    v_reports = [
        validator.validate_dataset(gold_df, "gold", target_date),
        validator.validate_dataset(silver_df, "silver", target_date),
        validator.validate_dataset(dxy_df, "dxy", target_date),
        validator.validate_dataset(us10y_df, "us10y", target_date),
        validator.validate_dataset(vix_df, "vix", target_date),
        validator.validate_dataset(oil_df, "oil", target_date),
        validator.validate_dataset(usdinr_df, "usdinr", target_date)
    ]
    q_report = validator.save_aggregated_quality_report(target_date, v_reports)

    # 3. Store raw & processed data
    storage.save_dataframe(gold_df, "processed/metals", "gold")
    storage.save_dataframe(silver_df, "processed/metals", "silver")
    storage.save_dataframe(dxy_df, "processed/macro", "dxy")
    storage.save_dataframe(us10y_df, "processed/macro", "us10y")

    # 4. Indian market unit conversion
    latest_gold_price = float(gold_df.iloc[-1]["close"])
    latest_silver_price = float(silver_df.iloc[-1]["close"])
    latest_usdinr = float(usdinr_df.iloc[-1]["close"]) if not usdinr_df.empty else 83.0

    inr_metrics = convert_usd_to_inr_metrics(latest_gold_price, latest_silver_price, latest_usdinr)

    # 5. Calculate technical indicators
    gold_tech = compute_all_technical_indicators(gold_df)
    silver_tech = compute_all_technical_indicators(silver_df)

    # 6. Cross-market correlation analysis
    corr_analyzer = MultiAssetCorrelationAnalyzer(windows=config.get("technical_indicators", {}).get("correlation_periods", [20, 60, 120, 252]))
    g_ret = gold_tech.set_index("date")["close"].pct_change()
    d_ret = dxy_df.set_index("date")["close"].pct_change()
    corrs = corr_analyzer.compute_correlations(g_ret, {"DXY": d_ret})

    # 7. Gold/Silver Ratio Relative Value
    ratio_analyzer = GoldSilverRatioAnalyzer()
    gsr_res = ratio_analyzer.analyze_ratio(gold_df, silver_df)

    # 8. Regimes classification
    macro_classifier = MacroRegimeClassifier()
    dxy_20d = float(d_ret.tail(20).sum()) if not d_ret.empty else 0.0
    us10y_change = float(us10y_df.iloc[-1]["close"] - us10y_df.iloc[-20]["close"]) if len(us10y_df) >= 20 else 0.0
    vix_level = float(vix_df.iloc[-1]["close"]) if not vix_df.empty else 15.0

    macro_res = macro_classifier.classify_macro_regime(
        dxy_return_20d=dxy_20d,
        us10y_change_20d=us10y_change,
        oil_return_20d=0.0,
        sp500_return_20d=0.0,
        vix_level=vix_level
    )

    market_classifier = MarketRegimeClassifier()
    latest_g = gold_tech.iloc[-1]
    g_regime = market_classifier.classify_market_regime(
        close_price=latest_g["close"],
        sma_20=latest_g["sma_20"],
        sma_50=latest_g["sma_50"],
        sma_200=latest_g["sma_200"],
        rsi_14=latest_g["rsi_14"],
        hist_vol_20=latest_g["hist_vol_20"]
    )

    latest_s = silver_tech.iloc[-1]
    s_regime = market_classifier.classify_market_regime(
        close_price=latest_s["close"],
        sma_20=latest_s["sma_20"],
        sma_50=latest_s["sma_50"],
        sma_200=latest_s["sma_200"],
        rsi_14=latest_s["rsi_14"],
        hist_vol_20=latest_s["hist_vol_20"]
    )

    # 9. Build ML feature store
    feature_builder = DailyFeatureStoreBuilder()
    features_df = feature_builder.build_feature_dataset(gold_df, silver_df, dxy_df, us10y_df, vix_df, oil_df)
    storage.save_dataframe(features_df, "features/daily", f"features_{target_date}")

    # 10. Historical Analogue Engine
    analogue_engine = HistoricalAnalogueEngine(
        top_k=config.get("analogue_engine", {}).get("top_k_analogues", 10),
        feature_weights=config.get("analogue_engine", {}).get("feature_weights")
    )
    cur_state = {
        "rsi_14": float(latest_g["rsi_14"]),
        "dist_sma_20": float(latest_g["dist_sma_20"]),
        "dist_sma_50": float(latest_g["dist_sma_50"]),
        "hist_vol_20": float(latest_g["hist_vol_20"]),
        "gold_silver_ratio": float(gsr_res.get("current_ratio", 80.0)),
        "dxy_return_20d": dxy_20d,
        "us10y_change_20d": us10y_change
    }
    analogues_res = analogue_engine.find_analogues(cur_state, features_df.iloc[:-1])

    # 11. Scores & Alerts
    scoring = ScoringSystem()
    g_scores = scoring.calculate_scores(
        rsi_14=float(latest_g["rsi_14"]),
        dist_sma_20=float(latest_g["dist_sma_20"]),
        dist_sma_200=float(latest_g["dist_sma_200"]),
        macd_hist=float(latest_g["macd_histogram"]),
        macro_regime=macro_res["regime"],
        gsr_zscore=gsr_res.get("z_score", 0.0),
        analogue_pos_prob=analogues_res.get("forward_statistics", {}).get("10d", {}).get("prob_positive_pct", 50.0),
        hist_vol_20=float(latest_g["hist_vol_20"])
    )

    detector = AlertDetector()
    alerts_res = detector.detect_alerts(
        date_str=target_date,
        gold_data={"rsi_14": float(latest_g["rsi_14"])},
        silver_data={"rsi_14": float(latest_s["rsi_14"])},
        gsr_data=gsr_res,
        macro_data=macro_res
    )

    # 12. Forecast Confidence & Predictions
    confidence_framework = PredictionConfidenceFramework()
    conf_res = confidence_framework.calculate_confidence(
        analogue_sample_size=analogues_res.get("top_k", 0),
        analogue_positive_prob=analogues_res.get("forward_statistics", {}).get("10d", {}).get("prob_positive_pct", 50.0),
        model_agreement_pct=0.75,
        macro_alignment=True,
        current_volatility=float(latest_g["hist_vol_20"])
    )

    predictions_data = {"gold": {"horizons": {}}}
    forecasts_summary = {}
    for h_str in ["1d", "3d", "5d", "10d", "20d", "60d"]:
        f_stat = analogues_res.get("forward_statistics", {}).get(h_str, {
            "prob_positive_pct": 50.0, "expected_mean_return_pct": 0.0, "median_return_pct": 0.0
        })
        f_stat["confidence_score"] = conf_res["confidence_score"]
        forecasts_summary[f"gold_{h_str}"] = f_stat

        pred_dir = "Bullish" if f_stat.get("prob_positive_pct", 50.0) >= 50.0 else "Bearish"
        predictions_data["gold"]["horizons"][h_str] = {
            "predicted_direction": pred_dir,
            "confidence_score": conf_res["confidence_score"]
        }

    tracker = ForecastPerformanceTracker()
    tracker.log_daily_predictions(target_date, predictions_data)
    eval_summary = tracker.evaluate_past_predictions(gold_df)

    # 13. Generate Daily Reports
    report_gen = ReportGenerator()
    report_payload = {
        "date": target_date,
        "gold": {
            "price": latest_gold_price,
            "return_1d_pct": float(gold_tech.iloc[-1]["close"] / gold_tech.iloc[-2]["close"] - 1) * 100,
            "return_5d_pct": float(gold_tech.iloc[-1]["close"] / gold_tech.iloc[-6]["close"] - 1) * 100,
            "return_20d_pct": float(gold_tech.iloc[-1]["close"] / gold_tech.iloc[-21]["close"] - 1) * 100,
            "market_regime": g_regime["composite_regime"],
            "research_bias": "Bullish" if g_scores["composite_score"] > 60 else ("Bearish" if g_scores["composite_score"] < 40 else "Neutral"),
            "rsi_14": float(latest_g["rsi_14"]),
            "macd": float(latest_g["macd"]),
            "dist_sma_200": float(latest_g["dist_sma_200"]),
            "sma_20": float(latest_g["sma_20"]),
            "sma_200": float(latest_g["sma_200"]),
            "scores": g_scores
        },
        "silver": {
            "price": latest_silver_price,
            "return_1d_pct": float(silver_tech.iloc[-1]["close"] / silver_tech.iloc[-2]["close"] - 1) * 100,
            "return_5d_pct": float(silver_tech.iloc[-1]["close"] / silver_tech.iloc[-6]["close"] - 1) * 100,
            "return_20d_pct": float(silver_tech.iloc[-1]["close"] / silver_tech.iloc[-21]["close"] - 1) * 100,
            "market_regime": s_regime["composite_regime"],
            "research_bias": "Neutral",
            "rsi_14": float(latest_s["rsi_14"]),
            "macd": float(latest_s["macd"]),
            "scores": {"composite_score": 50.0}
        },
        "indian_market": inr_metrics,
        "macro": macro_res,
        "gold_silver_ratio": gsr_res,
        "historical_analogues": analogues_res,
        "forecasts": forecasts_summary,
        "data_quality": q_report
    }

    report_gen.generate_json_report(target_date, report_payload)
    md_path = report_gen.generate_markdown_report(target_date, report_payload)

    logger.info(f"Pipeline complete for {target_date}. Daily report generated at: {md_path}")
    return report_payload

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Precious Metals Pipeline")
    parser.add_argument("date", nargs="?", default=datetime.now().strftime("%Y-%m-%d"), help="Target date YYYY-MM-DD")
    args = parser.parse_args()

    run_pipeline(args.date)
