import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates human-readable Markdown research reports and machine-readable JSON outputs."""

    @staticmethod
    def generate_markdown_report(
        date_str: str,
        analysis_data: Dict[str, Any],
        output_filepath: str,
    ) -> str:
        """
        Generates reports/daily/YYYY-MM-DD.md
        """
        gold = analysis_data.get("gold", {})
        silver = analysis_data.get("silver", {})
        macro = analysis_data.get("macro", {})
        ratio = analysis_data.get("gold_silver_ratio", {})
        analogues = analysis_data.get("analogues", {})
        forecasts = analysis_data.get("forecasts", {})
        alerts = analysis_data.get("alerts", {})
        quality = analysis_data.get("data_quality", {})

        md = f"""# Precious Metals Daily Research Report — {date_str}

## Executive Summary

**Gold Spot (USD/oz):**
* Price: ${gold.get('price', 0.0):,.2f}
* Daily Change: {gold.get('daily_change_pct', 0.0):+.2f}%
* 20d Volatility: {gold.get('volatility_20d', 0.0)*100:.1f}%
* Trend Regime: {gold.get('trend_regime', 'N/A')}
* Research Score: {gold.get('composite_score', 50.0)}/100

**Silver Spot (USD/oz):**
* Price: ${silver.get('price', 0.0):,.2f}
* Daily Change: {silver.get('daily_change_pct', 0.0):+.2f}%
* Trend Regime: {silver.get('trend_regime', 'N/A')}

---

## Gold Analysis

* **Technical Structure:** RSI(14) is {gold.get('rsi_14', 50.0):.1f}, MACD Histogram is {gold.get('macd_hist', 0.0):+.2f}.
* **Distance to 200 DMA:** {gold.get('dist_sma_200', 0.0):+.2f}%
* **Macro Environment:** Classified as `{macro.get('primary_regime', 'Neutral')}`.

---

## Silver Analysis & Gold/Silver Ratio

* **Gold/Silver Ratio:** {ratio.get('current_ratio', 0.0):.2f} (Z-Score: {ratio.get('z_score', 0.0):+.2f}, Percentile: {ratio.get('percentile', 50.0):.1f}%)
* **Relative Value Signal:** { "Ratio extreme high — Silver historical mean-reversion opportunity" if ratio.get('is_extreme_high') else ("Ratio extreme low — Gold relative value opportunity" if ratio.get('is_extreme_low') else "Ratio within normal historical boundaries") }

---

## Macro Environment & Cross-Market Drivers

* **Macro Regime:** {macro.get('primary_regime', 'Neutral')}
* **Explanation:** {macro.get('explainability', 'N/A')}

---

## Historical Analogue Analysis

Based on multi-factor similarity matching against historical market conditions:
* **Analogue Sample Count:** {analogues.get('top_k', 0)} matches found.
* **10-Day Historical Forward Win Probability:** {analogues.get('forward_stats', {}).get('10d', {}).get('win_probability', 50.0):.1f}%
* **10-Day Median Return:** {analogues.get('forward_stats', {}).get('10d', {}).get('median_return_pct', 0.0):+.2f}%

---

## Probabilistic Forecast Research Signals

*Disclaimer: Probabilistic research signals based on historical conditional statistics and statistical models. Not guaranteed future price outcomes.*

| Horizon | Positive Prob | Expected Return | Expected Volatility | Max Gain (Hist) | Max Loss (Hist) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1 Day** | {forecasts.get('1d', {}).get('probability_positive', 0.5)*100:.1f}% | {forecasts.get('1d', {}).get('expected_return_pct', 0.0):+.2f}% | {forecasts.get('1d', {}).get('expected_volatility_pct', 0.0):.2f}% | {forecasts.get('1d', {}).get('historical_max_gain_pct', 0.0):+.2f}% | {forecasts.get('1d', {}).get('historical_max_loss_pct', 0.0):+.2f}% |
| **5 Days** | {forecasts.get('5d', {}).get('probability_positive', 0.5)*100:.1f}% | {forecasts.get('5d', {}).get('expected_return_pct', 0.0):+.2f}% | {forecasts.get('5d', {}).get('expected_volatility_pct', 0.0):.2f}% | {forecasts.get('5d', {}).get('historical_max_gain_pct', 0.0):+.2f}% | {forecasts.get('5d', {}).get('historical_max_loss_pct', 0.0):+.2f}% |
| **10 Days** | {forecasts.get('10d', {}).get('probability_positive', 0.5)*100:.1f}% | {forecasts.get('10d', {}).get('expected_return_pct', 0.0):+.2f}% | {forecasts.get('10d', {}).get('expected_volatility_pct', 0.0):.2f}% | {forecasts.get('10d', {}).get('historical_max_gain_pct', 0.0):+.2f}% | {forecasts.get('10d', {}).get('historical_max_loss_pct', 0.0):+.2f}% |
| **20 Days** | {forecasts.get('20d', {}).get('probability_positive', 0.5)*100:.1f}% | {forecasts.get('20d', {}).get('expected_return_pct', 0.0):+.2f}% | {forecasts.get('20d', {}).get('expected_volatility_pct', 0.0):.2f}% | {forecasts.get('20d', {}).get('historical_max_gain_pct', 0.0):+.2f}% | {forecasts.get('20d', {}).get('historical_max_loss_pct', 0.0):+.2f}% |
| **60 Days** | {forecasts.get('60d', {}).get('probability_positive', 0.5)*100:.1f}% | {forecasts.get('60d', {}).get('expected_return_pct', 0.0):+.2f}% | {forecasts.get('60d', {}).get('expected_volatility_pct', 0.0):.2f}% | {forecasts.get('60d', {}).get('historical_max_gain_pct', 0.0):+.2f}% | {forecasts.get('60d', {}).get('historical_max_loss_pct', 0.0):+.2f}% |

---

## Active Alerts ({alerts.get('alerts_count', 0)})

{ chr(10).join([f"- **[{a['severity']}] {a['type']}**: {a['message']}" for a in alerts.get('alerts', [])]) if alerts.get('alerts') else "No active threshold alerts." }

---

## Data Quality Summary

* **Instruments Validated:** {quality.get('instruments_validated', 0)}
* **Errors:** {len(quality.get('summaries', [{}])[0].get('errors', [])) if quality.get('summaries') else 0}
* **Warnings:** {len(quality.get('summaries', [{}])[0].get('warnings', [])) if quality.get('summaries') else 0}
"""

        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        with open(output_filepath, "w") as f:
            f.write(md.strip())

        logger.info(f"Generated Markdown report at {output_filepath}")
        return md

    @staticmethod
    def generate_json_report(
        date_str: str,
        analysis_data: Dict[str, Any],
        output_filepath: str,
    ) -> Dict[str, Any]:
        """Generates data/analysis/daily/YYYY-MM-DD.json"""
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        with open(output_filepath, "w") as f:
            json.dump(analysis_data, f, indent=2)
        logger.info(f"Saved machine-readable JSON analysis report to {output_filepath}")
        return analysis_data
