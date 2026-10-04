import os
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class AlertEngine:
    """Detects significant technical, macro, or relative value events based on configurable thresholds."""

    @staticmethod
    def detect_alerts(
        date_str: str,
        gold_df_row: Any,
        silver_df_row: Any,
        ratio_df_row: Any,
        macro_dict: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect and produce alert list."""
        alerts_config = config.get("alerts", {})
        jump_thresh = float(alerts_config.get("price_jump_pct", 3.5))
        rsi_ub = float(alerts_config.get("rsi_overbought", 70.0))
        rsi_lb = float(alerts_config.get("rsi_oversold", 30.0))
        ratio_z_thresh = float(alerts_config.get("ratio_zscore_extreme", 2.0))

        detected = []

        # Gold checks
        if gold_df_row is not None:
            g_close = float(gold_df_row.get("close", 0))
            g_ret_1d = float(gold_df_row.get("return_1d", 0)) * 100.0 if "return_1d" in gold_df_row else 0.0
            g_rsi = float(gold_df_row.get("rsi_14", 50.0))

            if abs(g_ret_1d) >= jump_thresh:
                detected.append({
                    "severity": "HIGH",
                    "type": "LARGE_PRICE_MOVE",
                    "instrument": "GOLD",
                    "message": f"Gold experienced a significant 1-day move of {round(g_ret_1d, 2)}%"
                })

            if g_rsi >= rsi_ub:
                detected.append({
                    "severity": "MEDIUM",
                    "type": "EXTREME_RSI",
                    "instrument": "GOLD",
                    "message": f"Gold RSI is overbought at {round(g_rsi, 2)}"
                })
            elif g_rsi <= rsi_lb:
                detected.append({
                    "severity": "MEDIUM",
                    "type": "EXTREME_RSI",
                    "instrument": "GOLD",
                    "message": f"Gold RSI is oversold at {round(g_rsi, 2)}"
                })

            if bool(gold_df_row.get("breakout_20d", False)):
                detected.append({
                    "severity": "HIGH",
                    "type": "BREAKOUT",
                    "instrument": "GOLD",
                    "message": "Gold closed at a 20-day high (Breakout)"
                })

            if bool(gold_df_row.get("breakdown_20d", False)):
                detected.append({
                    "severity": "HIGH",
                    "type": "BREAKDOWN",
                    "instrument": "GOLD",
                    "message": "Gold closed at a 20-day low (Breakdown)"
                })

        # Ratio checks
        if ratio_df_row is not None:
            ratio_z = float(ratio_df_row.get("ratio_zscore", 0.0))
            if abs(ratio_z) >= ratio_z_thresh:
                severity = "HIGH" if abs(ratio_z) >= 2.5 else "MEDIUM"
                detected.append({
                    "severity": severity,
                    "type": "RATIO_EXTREME",
                    "instrument": "GOLD_SILVER_RATIO",
                    "message": f"Gold/Silver Ratio reached extreme Z-score of {round(ratio_z, 2)}"
                })

        return {
            "date": date_str,
            "alert_count": len(detected),
            "alerts": detected
        }


class ResearchReportGenerator:
    """Generates daily Markdown research reports and machine-readable JSON outputs."""

    @staticmethod
    def generate_markdown_report(
        date_str: str,
        analysis_data: Dict[str, Any],
        output_dir: str = "reports/daily"
    ) -> str:
        """Write structured daily Markdown report matching required specification."""
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{date_str}.md")

        gold_info = analysis_data.get("gold", {})
        silver_info = analysis_data.get("silver", {})
        macro_info = analysis_data.get("macro", {})
        ratio_info = analysis_data.get("gold_silver_ratio", {})
        quality_info = analysis_data.get("data_quality", {})

        md = f"""# Precious Metals Daily Research Report
**Date:** {date_str}

## Executive Summary

### Gold
* **Current Price:** ${gold_info.get('price', 0.0):,.2f} USD
* **Daily Move:** {gold_info.get('return_1d_pct', 0.0):+.2f}%
* **Weekly Move:** {gold_info.get('return_5d_pct', 0.0):+.2f}%
* **Monthly Move:** {gold_info.get('return_20d_pct', 0.0):+.2f}%
* **Trend:** {gold_info.get('trend_regime', 'N/A')}
* **Volatility:** {gold_info.get('volatility_regime', 'N/A')}
* **Research Bias:** Composite Score {gold_info.get('composite_score', 50.0):.1f}/100 ({gold_info.get('research_bias', 'Neutral')})

### Silver
* **Current Price:** ${silver_info.get('price', 0.0):,.2f} USD
* **Daily Move:** {silver_info.get('return_1d_pct', 0.0):+.2f}%
* **Weekly Move:** {silver_info.get('return_5d_pct', 0.0):+.2f}%
* **Monthly Move:** {silver_info.get('return_20d_pct', 0.0):+.2f}%
* **Trend:** {silver_info.get('trend_regime', 'N/A')}
* **Volatility:** {silver_info.get('volatility_regime', 'N/A')}
* **Research Bias:** Composite Score {silver_info.get('composite_score', 50.0):.1f}/100 ({silver_info.get('research_bias', 'Neutral')})

---

## Gold Analysis
* **Technical Structure:** RSI={gold_info.get('rsi_14', 50.0):.1f}, MACD={gold_info.get('macd', 0.0):.2f}, Distance 200DMA={gold_info.get('dist_sma_200_pct', 0.0):+.2f}%
* **Macro Drivers:** Sensitivity to DXY ({macro_info.get('explainable_components', {}).get('dxy_return_20d', 0.0):+.2f}% 20D move) and Rates
* **Key Support:** ${gold_info.get('sma_50', 0.0):,.2f} (50 DMA) / ${gold_info.get('sma_200', 0.0):,.2f} (200 DMA)
* **Key Resistance:** ${gold_info.get('bb_upper', 0.0):,.2f} (Upper BB)
* **Historical Analogues:** Top similarity match from historical states found {gold_info.get('analogue_sample_count', 0)} matches
* **Forward Probabilities (5D):** Positive Return Probability {gold_info.get('signals', {}).get('5d', {}).get('prob_positive', 0.5)*100:.1f}%

---

## Silver Analysis
* **Technical Structure:** RSI={silver_info.get('rsi_14', 50.0):.1f}, MACD={silver_info.get('macd', 0.0):.2f}
* **Macro Drivers:** Industrial demand vs precious metals monetary demand
* **Gold/Silver Ratio:** Current = {ratio_info.get('current_ratio', 0.0):.2f} (Z-score: {ratio_info.get('zscore', 0.0):+.2f})
* **Historical Analogues:** Sample count = {silver_info.get('analogue_sample_count', 0)}
* **Forward Probabilities (5D):** Positive Return Probability {silver_info.get('signals', {}).get('5d', {}).get('prob_positive', 0.5)*100:.1f}%

---

## Macro Environment
* **Macro Regime:** {macro_info.get('macro_regime', 'Neutral')}
* **DXY (US Dollar Index):** 20-Day Return = {macro_info.get('explainable_components', {}).get('dxy_return_20d', 0.0):+.2f}%
* **Treasury Yields (10Y):** 20-Day Change = {macro_info.get('explainable_components', {}).get('us10y_change_20d', 0.0):+.2f} pts
* **Volatility (VIX):** Current Level = {macro_info.get('explainable_components', {}).get('vix_level', 20.0):.1f}
* **Crude Oil:** 20-Day Return = {macro_info.get('explainable_components', {}).get('oil_return_20d', 0.0):+.2f}%

---

## Cross-Market Relationships & Correlation Regimes
Gold and Silver correlations against DXY and Treasury yields are continuously monitored. Stable negative correlation with DXY indicates strong currency-driven pricing.

---

## Historical Analogue Analysis
Current multi-factor state evaluated against historical precious metals database. Forward statistics derived strictly without look-ahead bias.

---

## Forecast Research (Probabilistic Signals)
*Note: All price predictions and forecasting outputs are strictly expressed as probabilistic research signals rather than guaranteed future prices.*

| Horizon | Gold Prob Positive | Gold Expected Return | Silver Prob Positive | Silver Expected Return |
| :--- | :--- | :--- | :--- | :--- |
| 1-Day | {gold_info.get('signals', {}).get('1d', {}).get('prob_positive', 0.5)*100:.1f}% | {gold_info.get('signals', {}).get('1d', {}).get('expected_return_pct', 0.0):+.2f}% | {silver_info.get('signals', {}).get('1d', {}).get('prob_positive', 0.5)*100:.1f}% | {silver_info.get('signals', {}).get('1d', {}).get('expected_return_pct', 0.0):+.2f}% |
| 5-Day | {gold_info.get('signals', {}).get('5d', {}).get('prob_positive', 0.5)*100:.1f}% | {gold_info.get('signals', {}).get('5d', {}).get('expected_return_pct', 0.0):+.2f}% | {silver_info.get('signals', {}).get('5d', {}).get('prob_positive', 0.5)*100:.1f}% | {silver_info.get('signals', {}).get('5d', {}).get('expected_return_pct', 0.0):+.2f}% |
| 20-Day | {gold_info.get('signals', {}).get('20d', {}).get('prob_positive', 0.5)*100:.1f}% | {gold_info.get('signals', {}).get('20d', {}).get('expected_return_pct', 0.0):+.2f}% | {silver_info.get('signals', {}).get('20d', {}).get('prob_positive', 0.5)*100:.1f}% | {silver_info.get('signals', {}).get('20d', {}).get('expected_return_pct', 0.0):+.2f}% |

---

## Risks
* Unanticipated Federal Reserve monetary policy shifts.
* Sudden spikes in USD / Treasury yields.
* Extreme geopolitical developments or liquidity events.

---

## Data Quality
* **Total Instruments Validated:** {quality_info.get('total_instruments_validated', 0)}
* **Errors Found:** {quality_info.get('total_errors', 0)}
* **Warnings / Suspicious Records:** {quality_info.get('total_warnings', 0)}
"""

        with open(filepath, "w") as f:
            f.write(md)

        return filepath
