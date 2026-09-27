import pandas as pd
import numpy as np
from typing import Dict, Any, List
from src.patterns.analogue_engine import HistoricalAnalogueEngine

class ForwardProbabilityEngine:
    """Generates probabilistic forward-looking signals for 1D, 3D, 5D, 10D, 20D, and 60D horizons."""

    def __init__(self, horizons: List[int] = [1, 3, 5, 10, 20, 60]):
        self.horizons = horizons
        self.analogue_engine = HistoricalAnalogueEngine(horizons=horizons)

    def generate_probabilistic_signals(
        self,
        df: pd.DataFrame,
        current_idx: int,
        instrument: str = "GOLD"
    ) -> Dict[str, Any]:
        """
        Generates conditional probabilistic forward statistics for an instrument based on historical analogues.
        Note: These are historical conditional research statistics, not guaranteed future outcomes.
        """
        price_col = f"{instrument.upper()}_close" if f"{instrument.upper()}_close" in df.columns else "GOLD_close"

        analogue_res = self.analogue_engine.find_analogues(
            df=df,
            current_idx=current_idx,
            price_col=price_col,
            top_n=15
        )

        fwd_stats = analogue_res.get("forward_stats", {})

        horizons_output = {}
        for h in self.horizons:
            key = f"{h}d"
            h_stat = fwd_stats.get(key, {})

            if h_stat:
                horizons_output[key] = {
                    "horizon_days": h,
                    "pos_prob": h_stat.get("pos_prob", 0.5),
                    "neg_prob": h_stat.get("neg_prob", 0.5),
                    "mean_return": h_stat.get("mean_return", 0.0),
                    "median_return": h_stat.get("median_return", 0.0),
                    "expected_volatility": h_stat.get("std", 0.0),
                    "max_gain": h_stat.get("max_gain", 0.0),
                    "max_loss": h_stat.get("max_loss", 0.0),
                    "sample_count": h_stat.get("count", 0)
                }
            else:
                horizons_output[key] = {
                    "horizon_days": h,
                    "pos_prob": 0.5,
                    "neg_prob": 0.5,
                    "mean_return": 0.0,
                    "median_return": 0.0,
                    "expected_volatility": 0.0,
                    "max_gain": 0.0,
                    "max_loss": 0.0,
                    "sample_count": 0
                }

        return {
            "disclaimer": "Probabilistic research signals based on conditional historical statistics. Not financial advice or guaranteed prices.",
            "instrument": instrument,
            "horizons": horizons_output,
            "sample_analogues_count": analogue_res.get("sample_count", 0)
        }
