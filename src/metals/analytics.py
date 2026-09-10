import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from src.indicators.technical import TechnicalIndicators

class GoldAnalyzer:
    """Dedicated analyzer for Gold multi-factor dynamics and macro relationships."""

    def analyze(self, gold_df: pd.DataFrame, macro_dfs: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, Any]:
        if gold_df.empty:
            return {}

        df = TechnicalIndicators.calculate_all(gold_df)
        latest = df.iloc[-1].to_dict()

        # Calculate rolling correlations with key macro drivers if available
        correlations = {}
        if macro_dfs:
            gold_close = df.set_index("date")["close"]
            for name, m_df in macro_dfs.items():
                if not m_df.empty and "close" in m_df.columns:
                    m_close = m_df.set_index("date")["close"]
                    combined = pd.concat([gold_close, m_close], axis=1, keys=["gold", name]).dropna()
                    if len(combined) >= 20:
                        correlations[f"{name}_20d"] = float(combined["gold"].rolling(20).corr(combined[name]).iloc[-1])
                    if len(combined) >= 60:
                        correlations[f"{name}_60d"] = float(combined["gold"].rolling(60).corr(combined[name]).iloc[-1])
                    if len(combined) >= 252:
                        correlations[f"{name}_252d"] = float(combined["gold"].rolling(252).corr(combined[name]).iloc[-1])

        price = float(latest.get("close", 0.0))
        dist_sma50 = float(latest.get("dist_sma_50", 0.0))
        rsi = float(latest.get("rsi_14", 50.0))

        technical_bias = "Neutral"
        if dist_sma50 > 0.02 and rsi > 55:
            technical_bias = "Bullish"
        elif dist_sma50 < -0.02 and rsi < 45:
            technical_bias = "Bearish"

        return {
            "symbol": "GOLD",
            "price": price,
            "daily_move_pct": float(latest.get("return_1d", 0.0)) * 100.0,
            "weekly_move_pct": float(latest.get("return_5d", 0.0)) * 100.0,
            "monthly_move_pct": float(latest.get("return_20d", 0.0)) * 100.0,
            "rsi_14": rsi,
            "macd": float(latest.get("macd", 0.0)),
            "macd_signal": float(latest.get("macd_signal", 0.0)),
            "sma_20": float(latest.get("sma_20", 0.0)),
            "sma_50": float(latest.get("sma_50", 0.0)),
            "sma_200": float(latest.get("sma_200", 0.0)),
            "volatility_20d": float(latest.get("volatility_20d", 0.0)),
            "technical_bias": technical_bias,
            "correlations": correlations
        }


class SilverAnalyzer:
    """Dedicated analyzer for Silver dynamics and ratio correlations."""

    def analyze(self, silver_df: pd.DataFrame, gold_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        if silver_df.empty:
            return {}

        df = TechnicalIndicators.calculate_all(silver_df)
        latest = df.iloc[-1].to_dict()

        gs_ratio_summary = {}
        if gold_df is not None and not gold_df.empty:
            g_close = gold_df.set_index("date")["close"]
            s_close = df.set_index("date")["close"]
            ratio = (g_close / s_close).dropna()
            if not ratio.empty:
                curr_ratio = float(ratio.iloc[-1])
                mean_ratio = float(ratio.mean())
                std_ratio = float(ratio.std()) if len(ratio) > 1 else 1.0
                z_score = float((curr_ratio - mean_ratio) / (std_ratio if std_ratio != 0 else 1.0))
                percentile = float((ratio < curr_ratio).mean() * 100.0)

                gs_ratio_summary = {
                    "current_ratio": curr_ratio,
                    "mean_ratio": mean_ratio,
                    "z_score": z_score,
                    "percentile": percentile
                }

        price = float(latest.get("close", 0.0))
        rsi = float(latest.get("rsi_14", 50.0))
        dist_sma50 = float(latest.get("dist_sma_50", 0.0))

        technical_bias = "Neutral"
        if dist_sma50 > 0.02 and rsi > 55:
            technical_bias = "Bullish"
        elif dist_sma50 < -0.02 and rsi < 45:
            technical_bias = "Bearish"

        return {
            "symbol": "SILVER",
            "price": price,
            "daily_move_pct": float(latest.get("return_1d", 0.0)) * 100.0,
            "weekly_move_pct": float(latest.get("return_5d", 0.0)) * 100.0,
            "monthly_move_pct": float(latest.get("return_20d", 0.0)) * 100.0,
            "rsi_14": rsi,
            "macd": float(latest.get("macd", 0.0)),
            "sma_20": float(latest.get("sma_20", 0.0)),
            "sma_50": float(latest.get("sma_50", 0.0)),
            "sma_200": float(latest.get("sma_200", 0.0)),
            "volatility_20d": float(latest.get("volatility_20d", 0.0)),
            "technical_bias": technical_bias,
            "gold_silver_ratio": gs_ratio_summary
        }


class RelativeValueAnalyzer:
    """Relative-value analysis between Gold and Silver."""

    def analyze(self, gold_df: pd.DataFrame, silver_df: pd.DataFrame) -> Dict[str, Any]:
        if gold_df.empty or silver_df.empty:
            return {}

        g_close = gold_df.set_index("date")["close"]
        s_close = silver_df.set_index("date")["close"]

        combined = pd.concat([g_close, s_close], axis=1, keys=["gold", "silver"]).dropna()
        if combined.empty:
            return {}

        ratio = combined["gold"] / combined["silver"]
        g_ret = combined["gold"].pct_change()
        s_ret = combined["silver"].pct_change()
        spread_ret = g_ret - s_ret

        curr_ratio = float(ratio.iloc[-1])
        rolling_mean_60 = float(ratio.rolling(60).mean().iloc[-1]) if len(ratio) >= 60 else float(ratio.mean())
        rolling_std_60 = float(ratio.rolling(60).std().iloc[-1]) if len(ratio) >= 60 else float(ratio.std())
        z_score_60 = (curr_ratio - rolling_mean_60) / (rolling_std_60 if rolling_std_60 != 0 else 1.0)

        # Extreme conditions
        extreme_high = curr_ratio > 85.0 or z_score_60 > 2.0
        extreme_low = curr_ratio < 65.0 or z_score_60 < -2.0

        return {
            "gold_silver_ratio": curr_ratio,
            "ratio_sma_60": rolling_mean_60,
            "ratio_zscore_60": float(z_score_60),
            "spread_return_1d": float(spread_ret.iloc[-1]) if not spread_ret.empty else 0.0,
            "spread_volatility_20d": float(spread_ret.rolling(20).std().iloc[-1] * np.sqrt(252)) if len(spread_ret) >= 20 else 0.0,
            "extreme_high_gold_overvalued": extreme_high,
            "extreme_low_silver_overvalued": extreme_low
        }
