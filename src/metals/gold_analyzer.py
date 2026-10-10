import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class GoldAnalyzer:
    """
    Dedicated analyzer for Gold (USD/oz, INR/10g, Futures).
    Calculates rolling correlations with DXY, US10Y, Oil, S&P 500, VIX, and USD/INR.
    """

    @staticmethod
    def analyze_gold(gold_df: pd.DataFrame, macro_dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        if gold_df.empty or "Close" not in gold_df.columns:
            return {}

        gold_close = gold_df["Close"]
        gold_ret = gold_close.pct_change()

        correlations = {}
        for name, mdf in macro_dfs.items():
            if not mdf.empty and "Close" in mdf.columns:
                m_ret = mdf["Close"].pct_change()
                combined = pd.concat([gold_ret, m_ret], axis=1).dropna()
                if len(combined) > 30:
                    correlations[f"corr_20d_{name}"] = float(combined.iloc[-20:, 0].corr(combined.iloc[-20:, 1]))
                    correlations[f"corr_60d_{name}"] = float(combined.iloc[-60:, 0].corr(combined.iloc[-60:, 1]))
                    correlations[f"corr_252d_{name}"] = float(combined.iloc[-252:, 0].corr(combined.iloc[-252:, 1]))

        latest_price = float(gold_close.iloc[-1])
        ret_1d = float((gold_close.iloc[-1] - gold_close.iloc[-2]) / gold_close.iloc[-2] * 100) if len(gold_close) > 1 else 0.0
        ret_20d = float((gold_close.iloc[-1] - gold_close.iloc[-20]) / gold_close.iloc[-20] * 100) if len(gold_close) >= 20 else 0.0
        ret_252d = float((gold_close.iloc[-1] - gold_close.iloc[-252]) / gold_close.iloc[-252] * 100) if len(gold_close) >= 252 else 0.0

        return {
            "latest_price": latest_price,
            "return_1d_pct": ret_1d,
            "return_20d_pct": ret_20d,
            "return_252d_pct": ret_252d,
            "correlations": correlations
        }
