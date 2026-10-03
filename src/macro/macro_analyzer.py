import pandas as pd
import numpy as np
from typing import Dict, Any, List


class MacroAnalyzer:
    """Analyzes cross-asset macroeconomic drivers, interest rates, equities, FX, and volatility."""

    def __init__(self):
        pass

    def analyze_macro_environment(
        self,
        dxy_df: pd.DataFrame,
        us10y_df: pd.DataFrame,
        us2y_df: pd.DataFrame,
        oil_df: pd.DataFrame,
        sp500_df: pd.DataFrame,
        nifty_df: pd.DataFrame,
        vix_df: pd.DataFrame,
        indiavix_df: pd.DataFrame,
        usdinr_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Compute latest level and return/change for all core macro series."""
        macro_summary = {}

        series_map = {
            "dxy": (dxy_df, "index"),
            "us10y": (us10y_df, "yield"),
            "us2y": (us2y_df, "yield"),
            "oil": (oil_df, "price"),
            "sp500": (sp500_df, "price"),
            "nifty50": (nifty_df, "price"),
            "vix": (vix_df, "index"),
            "indiavix": (indiavix_df, "index"),
            "usdinr": (usdinr_df, "fx"),
        }

        for name, (df, s_type) in series_map.items():
            if df is not None and not df.empty and "Close" in df.columns:
                latest_val = df["Close"].iloc[-1]
                prev_val = df["Close"].iloc[-2] if len(df) > 1 else latest_val

                if s_type in ["yield", "index"]:
                    abs_change = latest_val - prev_val
                    pct_change = (abs_change / prev_val) * 100.0 if prev_val != 0 else 0.0
                else:
                    abs_change = latest_val - prev_val
                    pct_change = (abs_change / prev_val) * 100.0 if prev_val != 0 else 0.0

                macro_summary[name] = {
                    "latest": round(float(latest_val), 4),
                    "abs_change": round(float(abs_change), 4),
                    "pct_change": round(float(pct_change), 4),
                    "type": s_type,
                }
            else:
                macro_summary[name] = {
                    "latest": None,
                    "abs_change": 0.0,
                    "pct_change": 0.0,
                    "type": s_type,
                }

        # Yield curve inversion check (US 10Y - US 2Y)
        if macro_summary["us10y"]["latest"] is not None and macro_summary["us2y"]["latest"] is not None:
            yield_spread = macro_summary["us10y"]["latest"] - macro_summary["us2y"]["latest"]
            macro_summary["yield_curve_inverted"] = yield_spread < 0
            macro_summary["yield_spread_10y_2y"] = round(yield_spread, 4)
        else:
            macro_summary["yield_curve_inverted"] = False
            macro_summary["yield_spread_10y_2y"] = 0.0

        return macro_summary
