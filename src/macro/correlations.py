import pandas as pd
import numpy as np
from typing import Dict, List, Any


class CrossMarketCorrelations:
    """Computes rolling multi-window correlations between precious metals and macro drivers."""

    @staticmethod
    def calculate_rolling_correlations(
        metal_df: pd.DataFrame,
        macro_dict: Dict[str, pd.DataFrame],
        windows: List[int] = [20, 60, 120, 252]
    ) -> pd.DataFrame:
        if metal_df is None or metal_df.empty:
            return pd.DataFrame()

        base = metal_df[["date", "close"]].rename(columns={"close": "metal_close"}).sort_values("date")
        base["metal_return"] = base["metal_close"].pct_change()

        merged = base.copy()

        # Merge macro assets
        for symbol, m_df in macro_dict.items():
            if m_df is None or m_df.empty or "close" not in m_df.columns:
                continue
            sub = m_df[["date", "close"]].rename(columns={"close": f"{symbol}_close"}).sort_values("date")
            sub[f"{symbol}_return"] = sub[f"{symbol}_close"].pct_change()
            merged = pd.merge(merged, sub, on="date", how="left")

        # Forward fill missing macro values up to 5 days
        macro_cols = [c for c in merged.columns if c not in ["date", "metal_close", "metal_return"]]
        merged[macro_cols] = merged[macro_cols].ffill(limit=5)

        res = merged[["date", "metal_close"]].copy()

        for symbol in macro_dict.keys():
            col_ret = f"{symbol}_return"
            if col_ret not in merged.columns:
                continue
            for w in windows:
                corr_col = f"corr_{symbol.lower()}_{w}d"
                res[corr_col] = merged["metal_return"].rolling(w, min_periods=max(5, w//4)).corr(merged[col_ret])

        return res

    @staticmethod
    def detect_correlation_regime_shifts(corr_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect significant changes or sign flips in rolling correlations."""
        shifts = []
        if corr_df is None or corr_df.empty or len(corr_df) < 20:
            return shifts

        latest = corr_df.iloc[-1]
        prev_20d = corr_df.iloc[-20] if len(corr_df) >= 20 else corr_df.iloc[0]

        for col in corr_df.columns:
            if col.startswith("corr_") and col.endswith("_60d"):
                symbol = col.split("_")[1].upper()
                curr_val = latest[col]
                prev_val = prev_20d[col]

                if pd.isna(curr_val) or pd.isna(prev_val):
                    continue

                # Detect sign flip or large shift (> 0.4 change)
                if (curr_val * prev_val < 0) or (abs(curr_val - prev_val) >= 0.4):
                    shifts.append({
                        "date": latest["date"],
                        "driver": symbol,
                        "metric": col,
                        "previous_60d_corr": round(float(prev_val), 3),
                        "current_60d_corr": round(float(curr_val), 3),
                        "description": f"Significant correlation shift with {symbol}: from {prev_val:.2f} to {curr_val:.2f}"
                    })

        return shifts
