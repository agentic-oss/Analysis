"""
Daily Markdown Report Generator.
Produces comprehensive reports/daily/YYYY-MM-DD.md
"""
import os
from pathlib import Path
from typing import Dict, Any


class DailyReportGenerator:
    """
    Generates Markdown Daily Precious Metals Research Report.
    """

    @staticmethod
    def generate_markdown_report(
        date_str: str,
        gold_analysis: Dict[str, Any],
        silver_analysis: Dict[str, Any],
        ratio_analysis: Dict[str, Any],
        macro_regime: Dict[str, Any],
        market_regimes: Dict[str, Any],
        gold_forecast: Dict[str, Any],
        silver_forecast: Dict[str, Any],
        quality_summary: Dict[str, Any],
        output_dir: str = "reports/daily",
    ) -> str:

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        report_path = Path(output_dir) / f"{date_str}.md"

        g_price = gold_analysis.get("price_usd_oz", 0.0)
        g_inr = gold_analysis.get("price_inr_10g", 0.0)
        g_ret1d = gold_analysis.get("return_1d_pct", 0.0)
        g_ret5d = gold_analysis.get("return_5d_pct", 0.0)
        g_bias = gold_forecast.get("research_bias", "Neutral")

        s_price = silver_analysis.get("price_usd_oz", 0.0)
        s_inr = silver_analysis.get("price_inr_kg", 0.0)
        s_ret1d = silver_analysis.get("return_1d_pct", 0.0)
        s_ret5d = silver_analysis.get("return_5d_pct", 0.0)
        s_bias = silver_forecast.get("research_bias", "Neutral")

        g_mkt_reg = market_regimes.get("GOLD", {}).get("composite_regime", "Neutral")
        s_mkt_reg = market_regimes.get("SILVER", {}).get("composite_regime", "Neutral")

        md_content = f"""# Precious Metals Daily Research Report — {date_str}

## Executive Summary

### Gold
* **Current Price (USD):** ${g_price:,.2f} / oz
* **Current Price (INR):** ₹{g_inr:,.2f} / 10g
* **1-Day Move:** {g_ret1d:+.2f}%
* **5-Day Move:** {g_ret5d:+.2f}%
* **Market Regime:** {g_mkt_reg}
* **Research Bias:** {g_bias}

### Silver
* **Current Price (USD):** ${s_price:,.2f} / oz
* **Current Price (INR):** ₹{s_inr:,.2f} / kg
* **1-Day Move:** {s_ret1d:+.2f}%
* **5-Day Move:** {s_ret5d:+.2f}%
* **Market Regime:** {s_mkt_reg}
* **Research Bias:** {s_bias}

---

## Gold Analysis
* **Technical Structure:** RSI(14): {gold_analysis.get('rsi_14', 50.0):.1f} | 200 DMA Dist: {gold_analysis.get('dist_sma_200_pct', 0.0):+.2f}%
* **Current Drawdown:** {gold_analysis.get('current_drawdown_pct', 0.0):.2f}%
* **50/200 DMA Crossover:** Golden Cross: {gold_analysis.get('golden_cross', False)} | Death Cross: {gold_analysis.get('death_cross', False)}

## Silver Analysis
* **Technical Structure:** RSI(14): {silver_analysis.get('rsi_14', 50.0):.1f} | 200 DMA Dist: {silver_analysis.get('dist_sma_200_pct', 0.0):+.2f}%
* **Current Drawdown:** {silver_analysis.get('current_drawdown_pct', 0.0):.2f}%
* **50/200 DMA Crossover:** Golden Cross: {silver_analysis.get('golden_cross', False)} | Death Cross: {silver_analysis.get('death_cross', False)}

---

## Gold/Silver Relative-Value Analysis
* **Current Ratio:** {ratio_analysis.get('current_ratio', 0.0):.2f}
* **252-Day Mean:** {ratio_analysis.get('mean_252d', 0.0):.2f}
* **Ratio Z-Score:** {ratio_analysis.get('zscore', 0.0):+.2f}
* **Relative Value Bias:** {ratio_analysis.get('relative_value_bias', 'Neutral')}

---

## Macro Environment
* **Primary Macro Regime:** {macro_regime.get('primary_regime', 'Neutral')}
* **Macro Description:** {macro_regime.get('description', '')}

---

## Probabilistic Forward Research Forecasts

> **Disclaimer:** All price predictions and forecasting outputs are framed strictly as historical conditional probabilistic research signals rather than guaranteed future prices.

### Gold Probabilities & Expected Returns
"""
        g_horizons = gold_forecast.get("horizons", {})
        for h_key, h_val in g_horizons.items():
            md_content += f"- **{h_key.upper()} Horizon:** Pos Prob: {h_val.get('probability_positive_return_pct', 50.0)}% | Median Expected Return: {h_val.get('expected_median_return_pct', 0.0):+.2f}%\n"

        md_content += "\n### Silver Probabilities & Expected Returns\n"
        s_horizons = silver_forecast.get("horizons", {})
        for h_key, h_val in s_horizons.items():
            md_content += f"- **{h_key.upper()} Horizon:** Pos Prob: {h_val.get('probability_positive_return_pct', 50.0)}% | Median Expected Return: {h_val.get('expected_median_return_pct', 0.0):+.2f}%\n"

        md_content += f"""
---

## Data Quality Summary
* **Total Valid Records:** {quality_summary.get('total_valid_records', 0)}
* **Suspicious Records:** {quality_summary.get('total_suspicious_records', 0)}
* **Errors Reported:** {len(quality_summary.get('errors', []))}
"""
        with open(report_path, "w") as f:
            f.write(md_content)

        return str(report_path)
