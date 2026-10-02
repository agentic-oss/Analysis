import numpy as np
import pandas as pd

class GoldMultiFactorAnalyzer:
    """Analyzes Gold relationships with macro drivers (DXY, Rates, Real Yields, Oil, Equities, VIX) and correlation regimes."""

    def __init__(self, windows: list[int] = None):
        self.windows = windows or [20, 60, 120, 252]

    def analyze_cross_market_correlations(self, gold_df: pd.DataFrame, macro_df: pd.DataFrame) -> dict:
        """
        Calculates rolling correlations for Gold returns against key macro assets over configured windows.
        gold_df must have columns ['date', 'close']
        macro_df must have columns ['date', 'symbol', 'close']
        """
        if gold_df.empty or macro_df.empty:
            return {}

        gold = gold_df.set_index("date")["close"].sort_index()
        gold_ret = gold.pct_change()

        # Pivot macro df
        macro_piv = macro_df.pivot(index="date", columns="symbol", values="close").sort_index()
        macro_rets = macro_piv.pct_change()

        results = {}

        for sym in macro_rets.columns:
            if sym == "GOLD":
                continue
            pair_results = {}
            ret_series = macro_rets[sym]
            combined = pd.concat([gold_ret, ret_series], axis=1, keys=["gold", sym]).dropna()

            if len(combined) < 20:
                continue

            for w in self.windows:
                if len(combined) >= w:
                    corr_series = combined["gold"].rolling(window=w).corr(combined[sym])
                    latest_corr = corr_series.iloc[-1]
                    pair_results[f"corr_{w}d"] = round(float(latest_corr), 4) if not np.isnan(latest_corr) else None

            results[sym] = pair_results

        # Determine overall correlation regime summary
        dxy_corr_20 = results.get("DXY", {}).get("corr_20d", 0.0) or 0.0
        yield_corr_20 = results.get("US10Y", {}).get("corr_20d", 0.0) or 0.0

        if dxy_corr_20 < -0.4 and yield_corr_20 < -0.4:
            regime = "Standard Macro Driver (Negative DXY/Yield Correlation)"
        elif dxy_corr_20 > 0.2 and yield_corr_20 > 0.2:
            regime = "Inflation / Safe-Haven Asset Allocation Shift"
        elif dxy_corr_20 < -0.4 and yield_corr_20 > 0.2:
            regime = "Currency Sensitivity Dominant"
        else:
            regime = "Mixed / Decoupled"

        return {
            "rolling_correlations": results,
            "correlation_regime": regime,
            "latest_dxy_correlation_20d": dxy_corr_20,
            "latest_yield_correlation_20d": yield_corr_20
        }
