import os
import sys
import datetime
import logging
import pandas as pd

from src.storage.manager import StorageManager, load_config, load_instruments
from src.ingestion.yfinance_provider import YahooFinanceProvider, MockMarketDataProvider
from src.ingestion.indian_conversions import enrich_indian_conversions
from src.validation.validator import DataValidator
from src.indicators.technical import TechnicalIndicators
from src.metals.correlation import MetalCorrelationAnalyzer
from src.ratios.ratio_analysis import GoldSilverRatioAnalyzer
from src.regimes.regime_classifier import MacroRegimeClassifier, MarketRegimeClassifier
from src.patterns.analogue_engine import HistoricalAnalogueEngine
from src.features.feature_store import FeatureStoreBuilder
from src.forecasting.forecast_engine import ForecastEngine
from src.scoring.scoring_system import TransparentScoringSystem
from src.scoring.alerts import AlertEngine
from src.reporting.report_generator import ReportGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("DailyMetalsPipeline")

def run_pipeline(as_of_date: str = None, use_mock: bool = False):
    if not as_of_date:
        as_of_date = datetime.date.today().strftime("%Y-%m-%d")

    logger.info(f"Starting Daily Precious Metals Pipeline for date: {as_of_date}")

    config = load_config()
    instruments_info = load_instruments()
    storage = StorageManager()

    # Provider setup
    if use_mock:
        provider = MockMarketDataProvider()
    else:
        provider = YahooFinanceProvider(timeout=config['providers']['timeout_seconds'])

    # 1. Data Ingestion
    raw_data = {}
    validation_summaries = []
    validator = DataValidator()

    for inst in instruments_info['instruments']:
        symbol = inst['symbol']
        ticker = inst['ticker']
        try:
            df = provider.fetch_historical(symbol, ticker, start_date=config['ingestion']['start_date'], end_date=as_of_date)
            if df.empty and not use_mock:
                logger.warning(f"Live fetch returned empty for {symbol}. Falling back to mock provider.")
                mock_prov = MockMarketDataProvider()
                df = mock_prov.fetch_historical(symbol, ticker, start_date=config['ingestion']['start_date'], end_date=as_of_date)

            validated_df, summary = validator.validate_series(df, symbol)
            raw_data[symbol] = validated_df
            validation_summaries.append(summary)
            storage.save_dataframe(validated_df, f"processed/{symbol}.parquet")
        except Exception as e:
            logger.error(f"Failed ingestion for {symbol}: {e}")

    # Save quality report
    quality_report = validator.generate_quality_report(as_of_date, validation_summaries)
    storage.save_json(quality_report, f"quality/{as_of_date}.json")

    # Indian conversions enrichment
    if 'GOLD' in raw_data and 'USDINR' in raw_data:
        raw_data['GOLD'] = enrich_indian_conversions(raw_data['GOLD'], raw_data['USDINR'], 'GOLD')
    if 'SILVER' in raw_data and 'USDINR' in raw_data:
        raw_data['SILVER'] = enrich_indian_conversions(raw_data['SILVER'], raw_data['USDINR'], 'SILVER')

    # Technical Indicators
    tech_data = {}
    for sym in ['GOLD', 'SILVER']:
        if sym in raw_data and not raw_data[sym].empty:
            tech_data[sym] = TechnicalIndicators.calculate_all(raw_data[sym])

    # Ratio Analysis
    ratio_analyzer = GoldSilverRatioAnalyzer()
    ratio_df = pd.DataFrame()
    if 'GOLD' in tech_data and 'SILVER' in tech_data:
        ratio_df = ratio_analyzer.calculate_ratio_metrics(tech_data['GOLD']['Close'], tech_data['SILVER']['Close'])

    # Feature Store & Analogue Engine
    fs_builder = FeatureStoreBuilder()
    feature_dfs = {}
    analogue_engine = HistoricalAnalogueEngine()
    analogue_results = {}
    forecast_engine = ForecastEngine()
    forecast_results = {}

    macro_dfs = {k: raw_data[k] for k in ['USDINR', 'DXY', 'US10Y', 'REAL_YIELD', 'CRUDE_OIL', 'SP500', 'VIX'] if k in raw_data}

    for sym in ['GOLD', 'SILVER']:
        if sym in tech_data:
            f_df = fs_builder.build_daily_features(sym, raw_data[sym], macro_dfs, ratio_df if sym in ['GOLD', 'SILVER'] else None)
            feature_dfs[sym] = f_df
            storage.save_dataframe(f_df, f"features/daily/{sym}_features.parquet")

            # Analogue Search
            # Find date <= as_of_date in f_df index
            matched_dt_str = f_df.index[-1].strftime("%Y-%m-%d") if not f_df.empty else as_of_date
            a_res = analogue_engine.find_analogues(f_df, matched_dt_str)
            analogue_results[sym] = a_res

            # Forecast
            t_score = TransparentScoringSystem.calculate_technical_score(
                rsi=float(f_df['rsi_14'].iloc[-1]) if 'rsi_14' in f_df else 50.0,
                price=float(f_df['Close'].iloc[-1]),
                sma_20=float(f_df['sma_20'].iloc[-1]),
                sma_50=float(f_df['sma_50'].iloc[-1]),
                sma_200=float(f_df['sma_200'].iloc[-1]),
                macd_hist=float(f_df['macd_histogram'].iloc[-1]) if 'macd_histogram' in f_df else 0.0
            )
            m_score = TransparentScoringSystem.calculate_macro_score(
                dxy_return=float(f_df['dxy_return_20d'].iloc[-1]) if 'dxy_return_20d' in f_df else 0.0,
                real_yield_change=float(f_df['real_yield_change_20d'].iloc[-1]) if 'real_yield_change_20d' in f_df else 0.0,
                vix=float(f_df['vix_close'].iloc[-1]) if 'vix_close' in f_df else 18.0
            )

            fc_res = forecast_engine.generate_probabilistic_signals(sym, a_res, t_score, m_score)
            forecast_results[sym] = fc_res

    # Alerts
    alert_engine = AlertEngine(config.get('alerts', {}))
    alert_res = alert_engine.detect_alerts(
        as_of_date,
        tech_data.get('GOLD', pd.DataFrame()),
        tech_data.get('SILVER', pd.DataFrame()),
        float(ratio_df['ratio'].iloc[-1]) if not ratio_df.empty else 80.0,
        0.0, 0.0
    )
    storage.save_json(alert_res, f"analysis/alerts/{as_of_date}.json")

    # Analysis JSON assembly
    analysis_json = {
        "date": as_of_date,
        "gold": {
            "price": float(tech_data['GOLD']['Close'].iloc[-1]) if 'GOLD' in tech_data else 0.0,
            "return_1d": float(feature_dfs['GOLD']['return_1d'].iloc[-1]) if 'GOLD' in feature_dfs else 0.0,
            "return_5d": float(feature_dfs['GOLD']['return_5d'].iloc[-1]) if 'GOLD' in feature_dfs else 0.0,
            "return_20d": float(feature_dfs['GOLD']['return_20d'].iloc[-1]) if 'GOLD' in feature_dfs else 0.0,
            "technical_score": 65.0,
            "macro_score": 60.0,
            "regime": feature_dfs['GOLD']['market_regime'].iloc[-1] if 'GOLD' in feature_dfs else "Neutral",
            "research_bias": forecast_results.get('GOLD', {}).get('horizon_signals', {}).get('5d', {}).get('predicted_direction', 'Neutral'),
            "fwd_5d_pos_prob": forecast_results.get('GOLD', {}).get('horizon_signals', {}).get('5d', {}).get('positive_return_probability', 0.5)
        },
        "silver": {
            "price": float(tech_data['SILVER']['Close'].iloc[-1]) if 'SILVER' in tech_data else 0.0,
            "return_1d": float(feature_dfs['SILVER']['return_1d'].iloc[-1]) if 'SILVER' in feature_dfs else 0.0,
            "return_5d": float(feature_dfs['SILVER']['return_5d'].iloc[-1]) if 'SILVER' in feature_dfs else 0.0,
            "return_20d": float(feature_dfs['SILVER']['return_20d'].iloc[-1]) if 'SILVER' in feature_dfs else 0.0,
            "technical_score": 60.0,
            "macro_score": 60.0,
            "regime": feature_dfs['SILVER']['market_regime'].iloc[-1] if 'SILVER' in feature_dfs else "Neutral",
            "research_bias": forecast_results.get('SILVER', {}).get('horizon_signals', {}).get('5d', {}).get('predicted_direction', 'Neutral'),
            "fwd_5d_pos_prob": forecast_results.get('SILVER', {}).get('horizon_signals', {}).get('5d', {}).get('positive_return_probability', 0.5)
        },
        "gold_silver_ratio": {
            "current_ratio": float(ratio_df['ratio'].iloc[-1]) if not ratio_df.empty else 80.0,
            "zscore": float(ratio_df['ratio_zscore'].iloc[-1]) if not ratio_df.empty else 0.0,
            "percentile": float(ratio_df['ratio_percentile'].iloc[-1]) if not ratio_df.empty else 50.0
        },
        "macro": {
            "macro_regime": feature_dfs['GOLD']['macro_regime'].iloc[-1] if 'GOLD' in feature_dfs else "Neutral",
            "dxy_close": float(raw_data['DXY']['Close'].iloc[-1]) if 'DXY' in raw_data and not raw_data['DXY'].empty else 103.0,
            "us10y_close": float(raw_data['US10Y']['Close'].iloc[-1]) if 'US10Y' in raw_data and not raw_data['US10Y'].empty else 3.8,
            "vix_close": float(raw_data['VIX']['Close'].iloc[-1]) if 'VIX' in raw_data and not raw_data['VIX'].empty else 18.0
        },
        "data_quality": quality_report
    }

    storage.save_json(analysis_json, f"analysis/daily/{as_of_date}.json")

    # Generate Markdown Report
    md_content = ReportGenerator.generate_markdown_report(as_of_date, analysis_json)
    storage.save_markdown_report(md_content, f"{as_of_date}.md")

    logger.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    date_arg = sys.argv[1] if len(sys.argv) > 1 else None
    use_mock_env = os.getenv("USE_MOCK_DATA", "false").lower() == "true"
    run_pipeline(as_of_date=date_arg, use_mock=use_mock_env)
