"""
Macro regime detection model incorporating composite cross-market signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


class MacroRegimeDetector:
    """
    Transparent rule-based Macro Regime classification.
    Possible macro regimes:
    - Inflationary
    - Disinflationary
    - Deflationary
    - Risk-On
    - Risk-Off
    - Tightening
    - Easing
    - Stagflationary
    - Neutral
    """

    @staticmethod
    def classify_row(row: pd.Series) -> Dict[str, Any]:
        """
        Classify macro regime for a single row of market indicators.
        Returns regime label and breakdown of underlying factors.
        """
        dxy_ret = row.get("dxy_return_20d", 0.0) or 0.0
        us10y_chg = row.get("us10y_change_20d", 0.0) or 0.0
        oil_ret = row.get("oil_return_20d", 0.0) or 0.0
        sp500_ret = row.get("sp500_return_20d", 0.0) or 0.0
        vix_val = row.get("vix_close", 20.0) or 20.0

        factors = {
            "dxy_return_20d": dxy_ret,
            "us10y_change_20d": us10y_chg,
            "oil_return_20d": oil_ret,
            "sp500_return_20d": sp500_ret,
            "vix": vix_val,
        }

        # Rule-based logic
        if oil_ret > 0.08 and us10y_chg > 0.15 and sp500_ret < -0.02:
            regime = "Stagflationary"
        elif oil_ret > 0.05 and us10y_chg > 0.10:
            regime = "Inflationary"
        elif vix_val > 25.0 or (sp500_ret < -0.05 and dxy_ret > 0.02):
            regime = "Risk-Off"
        elif sp500_ret > 0.03 and vix_val < 18.0 and us10y_chg <= 0.05:
            regime = "Risk-On"
        elif us10y_chg > 0.20 and dxy_ret > 0.02:
            regime = "Tightening"
        elif us10y_chg < -0.20 and dxy_ret < -0.02:
            regime = "Easing"
        elif oil_ret < -0.08 and us10y_chg < -0.10:
            regime = "Disinflationary"
        elif oil_ret < -0.12 and sp500_ret < -0.08:
            regime = "Deflationary"
        else:
            regime = "Neutral"

        return {
            "macro_regime": regime,
            "factors": factors
        }

    @classmethod
    def classify_df(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Classifies macro regime for an entire dataframe."""
        if df.empty:
            return df

        df = df.copy()
        regimes = []
        for idx, row in df.iterrows():
            res = cls.classify_row(row)
            regimes.append(res["macro_regime"])

        df["macro_regime"] = regimes
        return df
