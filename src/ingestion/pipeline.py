import logging
import datetime
from typing import Dict, List, Any, Optional
import pandas as pd

from src.ingestion.yfinance_provider import YFinanceProvider, TROY_OUNCE_TO_GRAMS, GRAMS_PER_10G, GRAMS_PER_KG

logger = logging.getLogger(__name__)

class DataIngestionPipeline:
    """Orchestrates market and macro data ingestion across categories with metadata enrichment."""

    def __init__(self, provider=None):
        self.provider = provider or YFinanceProvider()

    def fetch_all(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        results = {}
        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # 1. USD Spot & Futures Gold/Silver
        logger.info("Ingesting Precious Metals USD Spot & Futures...")
        gold_df = self.provider.get_spot_prices("GOLD", start_date, end_date)
        gold_df = self._enrich_metadata(gold_df, "GOLD", "spot", "USD", "oz", "LBMA")
        results["gold_usd"] = gold_df

        silver_df = self.provider.get_spot_prices("SILVER", start_date, end_date)
        silver_df = self._enrich_metadata(silver_df, "SILVER", "spot", "USD", "oz", "LBMA")
        results["silver_usd"] = silver_df

        # 2. Currency Data
        logger.info("Ingesting Currency & FX data...")
        usdinr_df = self.provider.get_currency_prices("USD/INR", start_date, end_date)
        usdinr_df = self._enrich_metadata(usdinr_df, "USD/INR", "fx", "INR", "ratio", "FX")
        results["usdinr"] = usdinr_df

        dxy_df = self.provider.get_currency_prices("DXY", start_date, end_date)
        dxy_df = self._enrich_metadata(dxy_df, "DXY", "index", "USD", "index", "ICE")
        results["dxy"] = dxy_df

        # 3. Indian Market Gold & Silver (First-Class Datasets & USD Conversions)
        logger.info("Calculating Indian Market Prices (Spot & MCX)...")
        results["gold_inr"] = self._compute_indian_gold(gold_df, usdinr_df)
        results["silver_inr"] = self._compute_indian_silver(silver_df, usdinr_df)

        mcx_gold_df = self.provider.get_historical_prices("GOLDBEES.NS", start_date, end_date)
        mcx_gold_df = self._enrich_metadata(mcx_gold_df, "MCX_GOLD", "futures", "INR", "10g", "MCX")
        results["mcx_gold"] = mcx_gold_df

        mcx_silver_df = self.provider.get_historical_prices("SILVERBEES.NS", start_date, end_date)
        mcx_silver_df = self._enrich_metadata(mcx_silver_df, "MCX_SILVER", "futures", "INR", "kg", "MCX")
        results["mcx_silver"] = mcx_silver_df

        # 4. Interest Rates & Macro Drivers
        logger.info("Ingesting Rates, Commodities, Equities, Volatility & Economic Events...")
        us10y = self.provider.get_macro_data("US10Y", start_date, end_date)
        results["us10y"] = self._enrich_metadata(us10y, "US10Y", "rates", "USD", "pct", "TREASURY")

        us2y = self.provider.get_macro_data("US2Y", start_date, end_date)
        results["us2y"] = self._enrich_metadata(us2y, "US2Y", "rates", "USD", "pct", "TREASURY")

        wti = self.provider.get_macro_data("OIL_WTI", start_date, end_date)
        results["oil_wti"] = self._enrich_metadata(wti, "OIL_WTI", "energy", "USD", "bbl", "NYMEX")

        sp500 = self.provider.get_macro_data("SP500", start_date, end_date)
        results["sp500"] = self._enrich_metadata(sp500, "SP500", "equity", "USD", "index", "S&P")

        nifty = self.provider.get_macro_data("NIFTY50", start_date, end_date)
        results["nifty50"] = self._enrich_metadata(nifty, "NIFTY50", "equity", "INR", "index", "NSE")

        vix = self.provider.get_macro_data("VIX", start_date, end_date)
        results["vix"] = self._enrich_metadata(vix, "VIX", "volatility", "USD", "index", "CBOE")

        india_vix = self.provider.get_macro_data("INDIA_VIX", start_date, end_date)
        results["india_vix"] = self._enrich_metadata(india_vix, "INDIA_VIX", "volatility", "INR", "index", "NSE")

        events = self.provider.get_economic_indicators(start_date, end_date)
        results["events"] = events

        return results

    def _enrich_metadata(self, df: pd.DataFrame, symbol: str, market: str, currency: str, unit: str, exchange: str) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df["provider"] = self.provider.name
        df["retrieval_timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        df["instrument"] = symbol
        df["market"] = market
        df["currency"] = currency
        df["unit"] = unit
        df["exchange"] = exchange
        df["timezone"] = "UTC"
        return df

    def _compute_indian_gold(self, gold_usd: pd.DataFrame, usdinr: pd.DataFrame) -> pd.DataFrame:
        if gold_usd.empty or usdinr.empty:
            return pd.DataFrame()
        merged = pd.merge(gold_usd[["date", "open", "high", "low", "close", "volume"]],
                          usdinr[["date", "close"]], on="date", suffixes=("", "_fx"))
        merged.rename(columns={"close_fx": "usdinr_rate"}, inplace=True)
        # Convert USD/oz -> INR/10g : (USD/oz * USDINR) / 31.1034768 * 10
        factor = (1.0 / TROY_OUNCE_TO_GRAMS) * GRAMS_PER_10G
        for col in ["open", "high", "low", "close"]:
            merged[f"inr_{col}"] = merged[col] * merged["usdinr_rate"] * factor
        merged["exchange_rate_used"] = merged["usdinr_rate"]
        merged["conversion_timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

        out = pd.DataFrame({
            "date": merged["date"],
            "open": merged["inr_open"],
            "high": merged["inr_high"],
            "low": merged["inr_low"],
            "close": merged["inr_close"],
            "volume": merged["volume"],
            "usd_price_orig": merged["close"],
            "exchange_rate_used": merged["exchange_rate_used"],
            "conversion_timestamp": merged["conversion_timestamp"]
        })
        return self._enrich_metadata(out, "GOLD_INR", "spot", "INR", "10g", "IBJA")

    def _compute_indian_silver(self, silver_usd: pd.DataFrame, usdinr: pd.DataFrame) -> pd.DataFrame:
        if silver_usd.empty or usdinr.empty:
            return pd.DataFrame()
        merged = pd.merge(silver_usd[["date", "open", "high", "low", "close", "volume"]],
                          usdinr[["date", "close"]], on="date", suffixes=("", "_fx"))
        merged.rename(columns={"close_fx": "usdinr_rate"}, inplace=True)
        # Convert USD/oz -> INR/kg : (USD/oz * USDINR) / 31.1034768 * 1000
        factor = (1.0 / TROY_OUNCE_TO_GRAMS) * GRAMS_PER_KG
        for col in ["open", "high", "low", "close"]:
            merged[f"inr_{col}"] = merged[col] * merged["usdinr_rate"] * factor
        merged["exchange_rate_used"] = merged["usdinr_rate"]
        merged["conversion_timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

        out = pd.DataFrame({
            "date": merged["date"],
            "open": merged["inr_open"],
            "high": merged["inr_high"],
            "low": merged["inr_low"],
            "close": merged["inr_close"],
            "volume": merged["volume"],
            "usd_price_orig": merged["close"],
            "exchange_rate_used": merged["exchange_rate_used"],
            "conversion_timestamp": merged["conversion_timestamp"]
        })
        return self._enrich_metadata(out, "SILVER_INR", "spot", "INR", "kg", "IBJA")
