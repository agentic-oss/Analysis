import datetime
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generates daily research Markdown report and structured JSON summary."""

    @staticmethod
    def generate_markdown_report(date_str: str, analysis_data: Dict[str, Any]) -> str:
        gold = analysis_data.get("gold", {})
        silver = analysis_data.get("silver", {})
        macro = analysis_data.get("macro", {})
        ratio = analysis_data.get("gold_silver_ratio", {})
        quality = analysis_data.get("data_quality", {})

        md = f"""# Precious Metals Daily Research Report
**Date:** {date_str}

## Executive Summary

### Gold
* **Current Price:** ${gold.get('price', 0.0):,.2f} USD
* **Daily Move:** {gold.get('return_1d', 0.0)*100:+.2f}%
* **Weekly Move:** {gold.get('return_5d', 0.0)*100:+.2f}%
* **Monthly Move:** {gold.get('return_20d', 0.0)*100:+.2f}%
* **Trend:** {gold.get('regime', 'N/A')}
* **Research Bias:** {gold.get('research_bias', 'Neutral')}

### Silver
* **Current Price:** ${silver.get('price', 0.0):,.2f} USD
* **Daily Move:** {silver.get('return_1d', 0.0)*100:+.2f}%
* **Weekly Move:** {silver.get('return_5d', 0.0)*100:+.2f}%
* **Monthly Move:** {silver.get('return_20d', 0.0)*100:+.2f}%
* **Trend:** {silver.get('regime', 'N/A')}
* **Research Bias:** {silver.get('research_bias', 'Neutral')}

---

## Gold Analysis
* **Technical Structure:** RSI: {gold.get('rsi_14', 50):.1f} | Dist 200 SMA: {gold.get('dist_sma_200', 0):+.2f}%
* **Composite Technical Score:** {gold.get('technical_score', 50)}/100
* **Composite Macro Score:** {gold.get('macro_score', 50)}/100
* **Key Support:** ${gold.get('price', 0.0)*0.97:,.2f} USD
* **Key Resistance:** ${gold.get('price', 0.0)*1.03:,.2f} USD

---

## Silver Analysis & Gold/Silver Ratio
* **Current Gold/Silver Ratio:** {ratio.get('current_ratio', 0.0):.2f}
* **Ratio Z-Score (252d):** {ratio.get('zscore', 0.0):+.2f}
* **Ratio Percentile:** {ratio.get('percentile', 50.0):.1f}%

---

## Macro Environment & Cross-Market Drivers
* **Macro Regime:** {macro.get('macro_regime', 'Neutral')}
* **US Dollar Index (DXY):** {macro.get('dxy_close', 'N/A')}
* **US 10Y Treasury Yield:** {macro.get('us10y_close', 'N/A')}%
* **CBOE Volatility Index (VIX):** {macro.get('vix_close', 'N/A')}

---

## Historical Analogue Analysis & Forward Probabilities
* **Gold 5-Day Forward Positive Return Probability:** {gold.get('fwd_5d_pos_prob', 0.5)*100:.1f}%
* **Silver 5-Day Forward Positive Return Probability:** {silver.get('fwd_5d_pos_prob', 0.5)*100:.1f}%

> *Disclaimer: All forecasting signals are historical conditional statistics, not guaranteed future outcomes.*

---

## Data Quality Summary
* **Total Records Analyzed:** {quality.get('total_records', 0)}
* **Warnings Count:** {len(quality.get('warnings', []))}
* **Errors Count:** {len(quality.get('errors', []))}
"""
        return md
