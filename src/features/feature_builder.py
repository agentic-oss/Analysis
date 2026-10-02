import pandas as pd
import numpy as np

class DailyFeatureBuilder:
    """Builds ML-ready daily feature store datasets with strict look-ahead protection and separate target horizons."""

    def build_feature_dataset(
        self,
        metal_df: pd.DataFrame,
        macro_df: pd.DataFrame,
        ratio_df: pd.DataFrame,
        macro_regime_dict: dict = None,
        market_regime_dict: dict = None
    ) -> pd.DataFrame:
        """
        Combines technical indicators, relative ratios, macro factors, and regimes into daily feature rows.
        Targets (`future_return_1d`, `future_return_3d`, `future_return_5d`, `future_return_10d`, `future_return_20d`, `future_return_60d`)
        are stored explicitly as forward target columns.
        """
        if metal_df.empty:
            return pd.DataFrame()

        df = metal_df.copy()

        # Core price returns
        df["return_1d"] = df["close"].pct_change(1)
        df["return_5d"] = df["close"].pct_change(5)
        df["return_20d"] = df["close"].pct_change(20)
        df["return_60d"] = df["close"].pct_change(60)

        # Distance from moving averages
        if "sma_200" in df.columns:
            df["distance_from_200dma"] = (df["close"] - df["sma_200"]) / df["sma_200"]

        # Merge Gold/Silver ratio metrics if available
        if ratio_df is not None and not ratio_df.empty and "ratio" in ratio_df.columns:
            r_subset = ratio_df[["ratio", "ratio_zscore"]].rename(
                columns={"ratio": "gold_silver_ratio", "ratio_zscore": "gold_silver_ratio_zscore"}
            ).reset_index()

            if "date" in df.columns and "date" in r_subset.columns:
                df["date"] = df["date"].astype(str)
                r_subset["date"] = r_subset["date"].astype(str)
                df = df.merge(r_subset, on="date", how="left")

        # Fill any missing ratio features with neutral defaults
        if "gold_silver_ratio" not in df.columns:
            df["gold_silver_ratio"] = np.nan
        if "gold_silver_ratio_zscore" not in df.columns:
            df["gold_silver_ratio_zscore"] = 0.0
        else:
            df["gold_silver_ratio_zscore"] = df["gold_silver_ratio_zscore"].fillna(0.0)

        # Merge macro factors if available
        if macro_df is not None and not macro_df.empty:
            piv = macro_df.pivot(index="date", columns="symbol", values="close")
            macro_rets = piv.pct_change(1)
            macro_rets_20 = piv.pct_change(20)

            df["date"] = df["date"].astype(str)
            if "DXY" in piv.columns:
                df["dxy_return"] = df["date"].map(macro_rets["DXY"]).fillna(0.0)
                df["dxy_return_20d"] = df["date"].map(macro_rets_20["DXY"]).fillna(0.0)
            if "US10Y" in piv.columns:
                df["us10y_change"] = df["date"].map(piv["US10Y"].diff(1)).fillna(0.0)
                df["us10y_change_20d"] = df["date"].map(piv["US10Y"].diff(20)).fillna(0.0)
            if "VIX" in piv.columns:
                df["vix_level"] = df["date"].map(piv["VIX"]).fillna(18.0)
                df["vix_change"] = df["date"].map(macro_rets["VIX"]).fillna(0.0)
            if "WTI" in piv.columns or "BRENT" in piv.columns:
                oil_col = "WTI" if "WTI" in piv.columns else "BRENT"
                df["oil_return"] = df["date"].map(macro_rets[oil_col]).fillna(0.0)
            if "SP500" in piv.columns:
                df["sp500_return"] = df["date"].map(macro_rets["SP500"]).fillna(0.0)
            if "NIFTY50" in piv.columns:
                df["nifty_return"] = df["date"].map(macro_rets["NIFTY50"]).fillna(0.0)

        # Ensure macro columns exist with defaults if missing
        for col, default_val in [
            ("dxy_return", 0.0), ("dxy_return_20d", 0.0),
            ("us10y_change", 0.0), ("us10y_change_20d", 0.0),
            ("vix_level", 18.0), ("vix_change", 0.0),
            ("oil_return", 0.0), ("sp500_return", 0.0), ("nifty_return", 0.0)
        ]:
            if col not in df.columns:
                df[col] = default_val

        # Attach regimes
        if macro_regime_dict:
            df["macro_regime"] = macro_regime_dict.get("primary_regime", "Neutral")
        if market_regime_dict:
            df["market_regime"] = market_regime_dict.get("trend_regime", "Range / Consolidation")

        # Separated Future Target Variables
        horizons = [1, 3, 5, 10, 20, 60]
        for h in horizons:
            df[f"future_return_{h}d"] = df["close"].pct_change(periods=h).shift(-h)
            df[f"future_direction_{h}d"] = (df[f"future_return_{h}d"] > 0).astype(float)

        return df
