import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class SilverAnalyzer:
    """
    Dedicated analyzer for Silver (USD/oz, INR/kg, Futures).
    Calculates rolling correlations with Gold, Industrial commodities, DXY, and Equities.
    """

    @staticmethod
    def analyze_silver(silver_df: pd.DataFrame, gold_df: pd.DataFrame, macro_dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        if silver_df.empty or "Close" not in silver_df.columns:
            return {}

        silver_close = silver_df["Close"]
        silver_ret = silver_close.pct_change()

        correlations = {}
        if not gold_df.empty and "Close" in gold_df.columns:
            g_ret = gold_df["Close"].pct_change()
            combined = pd.concat([silver_ret, g_ret], axis=1).dropna()
            if len(combined) > 30:
                correlations["corr_20d_GOLD"] = float(combined.iloc[-20:, 0].corr(combined.iloc[-20:, 1]))
                correlations["corr_60d_GOLD"] = float(combined.iloc[-60:, 0].corr(combined.iloc[-60:, 1]))
                correlations["corr_252d_GOLD"] = float(combined.iloc[-252:, 0].corr(combined.iloc[-252:, 1]))

        for name, mdf in macro_dfs.items():
            if not mdf.empty and "Close" in mdf.columns:
                m_ret = mdf["Close"].pct_change()
                combined = pd.concat([silver_ret, m_ret], axis=1).dropna()
                if len(combined) > 30:
                    correlations[f"corr_20d_{name}"] = float(combined.iloc[-20:, 0].corr(combined.iloc[-20:, 1]))

        latest_price = float(silver_close.iloc[-1])
        ret_1d = float((silver_close.iloc[-1] - silver_close.iloc[-2]) / silver_close.iloc[-2] * 100) if len(silver_close) > 1 else 0.0
        ret_20d = float((silver_close.iloc[-1] - silver_close.iloc[-20]) / silver_close.iloc[-20] * 100) if len(silver_close) >= 20 else 0.0

        return {
            "latest_price": latest_price,
            "return_1d_pct": ret_1d,
            "return_20d_pct": ret_20d,
            "correlations": correlations
        }
