import pandas as pd
import numpy as np
from typing import Dict, Any, List

class MultiAssetCorrelationAnalyzer:
    """Calculates rolling correlations across metals, rates, currency, equities, and commodities."""

    def __init__(self, windows: List[int] = [20, 60, 120, 252]):
        self.windows = windows

    def compute_correlations(
        self,
        base_returns: pd.Series,
        other_returns_dict: Dict[str, pd.Series]
    ) -> Dict[str, Dict[int, float]]:
        """
        Computes rolling correlation between base asset and dictionary of target assets over configured windows.
        Returns latest rolling correlation for each window and target asset.
        """
        results = {}
        for target_name, target_series in other_returns_dict.items():
            results[target_name] = {}
            aligned = pd.concat([base_returns, target_series], axis=1, keys=["base", "target"]).dropna()
            if len(aligned) < 5:
                for w in self.windows:
                    results[target_name][w] = 0.0
                continue

            for w in self.windows:
                min_p = min(w, max(5, len(aligned) // 2))
                corr_series = aligned["base"].rolling(window=w, min_periods=min_p).corr(aligned["target"])
                val = corr_series.iloc[-1]
                results[target_name][w] = round(float(val), 4) if not np.isnan(val) else 0.0

        return results

    def detect_regime_shifts(
        self,
        base_returns: pd.Series,
        target_returns: pd.Series,
        short_window: int = 20,
        long_window: int = 120,
        threshold: float = 0.4
    ) -> Dict[str, Any]:
        """Detects if short-term correlation significantly diverges from long-term baseline."""
        aligned = pd.concat([base_returns, target_returns], axis=1, keys=["base", "target"]).dropna()
        if len(aligned) < long_window:
            return {"shift_detected": False, "short_corr": 0.0, "long_corr": 0.0, "diff": 0.0}

        short_corr = aligned["base"].rolling(short_window).corr(aligned["target"]).iloc[-1]
        long_corr = aligned["base"].rolling(long_window).corr(aligned["target"]).iloc[-1]

        diff = short_corr - long_corr
        shift_detected = abs(diff) >= threshold

        return {
            "shift_detected": bool(shift_detected),
            "short_corr": round(float(short_corr), 4),
            "long_corr": round(float(long_corr), 4),
            "diff": round(float(diff), 4)
        }
