import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generates daily Markdown research report and structured JSON analysis."""

    def generate_markdown_report(self, date_str: str, pipeline_data: dict, output_dir: str = "reports/daily") -> str:
        """Builds formatted Markdown report following exact Section 23 specification."""
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{date_str}.md")

        gold = pipeline_data.get("gold", {})
        silver = pipeline_data.get("silver", {})
        ratio = pipeline_data.get("gold_silver_ratio", {})
        macro = pipeline_data.get("macro", {})
        analogues = pipeline_data.get("analogues", {})
        quality = pipeline_data.get("data_quality", {})
        alerts = pipeline_data.get("alerts", {})

        g_price = gold.get("price", 0.0)
        g_1d = gold.get("return_1d_pct", 0.0)
        g_5d = gold.get("return_5d_pct", 0.0)
        g_20d = gold.get("return_20d_pct", 0.0)
        g_regime = gold.get("regime", "N/A")
        g_score = gold.get("scores", {}).get("composite_score", 50.0)

        s_price = silver.get("price", 0.0)
        s_1d = silver.get("return_1d_pct", 0.0)
        s_5d = silver.get("return_5d_pct", 0.0)
        s_20d = silver.get("return_20d_pct", 0.0)
        s_regime = silver.get("regime", "N/A")
        s_score = silver.get("scores", {}).get("composite_score", 50.0)

        md = f"""# Precious Metals Daily Research Report — {date_str}

> **Research Disclaimer**: All price predictions and forecasting statistics strictly represent probabilistic research signals derived from historical analogues and quantitative models rather than guaranteed future outcomes.

---

## Executive Summary

### Gold Summary
* **Current Price**: ${g_price:,.2f} USD/oz
* **1-Day Move**: {g_1d:+.2f}%
* **1-Week Move**: {g_5d:+.2f}%
* **1-Month Move**: {g_20d:+.2f}%
* **Trend & Volatility Regime**: {g_regime}
* **Composite Score**: {g_score} / 100 ({gold.get('scores', {}).get('score_interpretation', 'Neutral')})
* **Research Bias**: {gold.get('research_bias', 'Neutral')}

### Silver Summary
* **Current Price**: ${s_price:,.2f} USD/oz
* **1-Day Move**: {s_1d:+.2f}%
* **1-Week Move**: {s_5d:+.2f}%
* **1-Month Move**: {s_20d:+.2f}%
* **Trend & Volatility Regime**: {s_regime}
* **Composite Score**: {s_score} / 100 ({silver.get('scores', {}).get('score_interpretation', 'Neutral')})
* **Research Bias**: {silver.get('research_bias', 'Neutral')}

---

## Gold Analysis

* **Technical Structure**: RSI 14 = {gold.get('rsi_14', 50.0):.1f}, MACD = {gold.get('macd', 0.0):.2f}, Distance from 200 DMA = {gold.get('dist_200dma', 0.0):+.2f}%
* **Key Support Level**: ${gold.get('support_level', g_price * 0.98):,.2f}
* **Key Resistance Level**: ${gold.get('resistance_level', g_price * 1.02):,.2f}
* **Macro Drivers**: DXY 20D Return = {macro.get('components', {}).get('dxy_return_20d', 0.0)*100:+.2f}%, US10Y 20D Yield Change = {macro.get('components', {}).get('us10y_change_20d', 0.0):+.2f}%

---

## Silver Analysis

* **Technical Structure**: RSI 14 = {silver.get('rsi_14', 50.0):.1f}, Distance from 200 DMA = {silver.get('dist_200dma', 0.0):+.2f}%
* **Gold / Silver Ratio**: {ratio.get('current_ratio', 0.0):.2f} (Z-Score: {ratio.get('ratio_zscore', 0.0):+.2f}, Percentile: {ratio.get('ratio_percentile', 50.0):.1f}%)
* **Relative Value Status**: {ratio.get('relative_value_regime', 'Neutral')}

---

## Macro Environment & Cross-Market Relationships

* **Macro Regime**: {macro.get('primary_regime', 'Neutral')}
* **Risk Stance**: {macro.get('risk_regime', 'Neutral')}
* **Monetary Stance**: {macro.get('monetary_regime', 'Neutral')}
* **VIX Level**: {macro.get('components', {}).get('vix_level', 0.0):.1f}
* **S&P 500 20D Return**: {macro.get('components', {}).get('sp500_return_20d', 0.0)*100:+.2f}%
* **Crude Oil 20D Return**: {macro.get('components', {}).get('oil_return_20d', 0.0)*100:+.2f}%

---

## Historical Analogue Analysis

Found {analogues.get('num_analogues', 0)} closest historical market analogues strictly prior to {date_str}.
* **Top Analogue Dates**: {', '.join(analogues.get('analogue_dates', [])[:5])}
* **10-Day Win Rate Frequency**: {analogues.get('forward_statistics', {}).get('10d', {}).get('win_rate_pct', 50.0):.1f}%
* **Expected 10-Day Mean Return**: {analogues.get('forward_statistics', {}).get('10d', {}).get('mean_return_pct', 0.0):+.2f}%

---

## Forecast Research (Probabilistic Signals)

### Gold Horizons
| Horizon | Win Probability | Expected Return | Signal | Confidence |
|---|---|---|---|---|
"""
        g_f = gold.get("forward_statistics", {}).get("forecasts", {})
        for h_key in ["1d", "3d", "5d", "10d", "20d", "60d"]:
            item = g_f.get(h_key, {})
            md += f"| {h_key.upper()} | {item.get('positive_return_probability', 0.5)*100:.0f}% | {item.get('expected_return_pct', 0.0):+.2f}% | {item.get('research_signal', 'Neutral')} | {item.get('confidence_score', 50)}/100 |\n"

        md += """
### Silver Horizons
| Horizon | Win Probability | Expected Return | Signal | Confidence |
|---|---|---|---|---|
"""
        s_f = silver.get("forward_statistics", {}).get("forecasts", {})
        for h_key in ["1d", "3d", "5d", "10d", "20d", "60d"]:
            item = s_f.get(h_key, {})
            md += f"| {h_key.upper()} | {item.get('positive_return_probability', 0.5)*100:.0f}% | {item.get('expected_return_pct', 0.0):+.2f}% | {item.get('research_signal', 'Neutral')} | {item.get('confidence_score', 50)}/100 |\n"

        md += f"""
---

## Key Thesis Risks & Invalidation Conditions
1. Sharp macro shift in US Treasury yields or USD strength.
2. Sudden geopolitical resolution or unexpected central bank policy surprise.
3. Breakdown of historical correlation parameters or extreme liquidity shocks.

---

## Data Quality Summary
* **Valid Records**: {quality.get('summary', {}).get('valid_count', 0)}
* **Suspicious Records**: {quality.get('summary', {}).get('suspicious_count', 0)}
* **Errors Reported**: {len(quality.get('errors', []))}
* **Alerts Triggered**: {alerts.get('alerts_count', 0)}
"""

        with open(filepath, "w") as f:
            f.write(md)

        return filepath
