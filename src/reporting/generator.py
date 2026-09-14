"""
Daily Markdown Research Report & Machine-Readable JSON Generator.
Produces:
  - reports/daily/YYYY-MM-DD.md
  - data/analysis/daily/YYYY-MM-DD.json
"""
import os
import json
import logging
from typing import Dict, List, Any
import pandas as pd

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates structured daily Markdown reports and machine-readable JSON files.
    """

    @staticmethod
    def generate_json_report(
        date_str: str,
        gold_analysis: Dict[str, Any],
        silver_analysis: Dict[str, Any],
        relative_value: Dict[str, Any],
        macro_regime: Dict[str, Any],
        market_regimes: Dict[str, Any],
        quality_report: Dict[str, Any],
        alerts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Builds schema-compliant machine-readable dictionary.
        """
        return {
            "date": date_str,
            "gold": gold_analysis,
            "silver": silver_analysis,
            "gold_silver_ratio": relative_value,
            "macro": macro_regime,
            "market_regime": market_regimes,
            "data_quality": quality_report,
            "alerts": alerts
        }

    @staticmethod
    def generate_markdown_report(
        date_str: str,
        gold_analysis: Dict[str, Any],
        silver_analysis: Dict[str, Any],
        relative_value: Dict[str, Any],
        macro_regime: Dict[str, Any],
        gold_regime: Dict[str, Any],
        silver_regime: Dict[str, Any],
        gold_analogues: Dict[str, Any],
        silver_analogues: Dict[str, Any],
        quality_report: Dict[str, Any],
        alerts: List[Dict[str, Any]]
    ) -> str:
        """
        Builds comprehensive Markdown report conforming to section 23 specifications.
        """
        g_price = gold_analysis.get("price", 0.0)
        g_score = gold_analysis.get("composite_score", 50.0)
        g_regime = gold_regime.get("composite_regime", "N/A")

        s_price = silver_analysis.get("price", 0.0)
        s_score = silver_analysis.get("composite_score", 50.0)
        s_regime = silver_regime.get("composite_regime", "N/A")

        ratio_val = relative_value.get("latest_ratio", 0.0)
        macro_reg = macro_regime.get("macro_regime", "Neutral")

        md = f"""# Precious Metals Daily Research Report — {date_str}

## Executive Summary

### Gold
* **Current Price:** ${g_price:,.2f} USD/oz
* **Market Regime:** {g_regime}
* **Research Factor Score:** {g_score:.1f} / 100
* **Research Bias (5D):** {gold_analysis.get("signals", {}).get("5d", {}).get("research_bias", "Neutral")}

### Silver
* **Current Price:** ${s_price:,.2f} USD/oz
* **Market Regime:** {s_regime}
* **Research Factor Score:** {s_score:.1f} / 100
* **Research Bias (5D):** {silver_analysis.get("signals", {}).get("5d", {}).get("research_bias", "Neutral")}

---

## Gold Analysis

* **Technical Structure:** Composite factor score {g_score:.1f}/100.
* **Macro Drivers:** Primary macro environment classified as **{macro_reg}**.
* **Historical Analogues:** Matched {gold_analogues.get("analogue_count", 0)} historical market conditions.
* **5-Day Forward Probability:** {gold_analysis.get("signals", {}).get("5d", {}).get("positive_return_probability_pct", 50.0)}% positive return frequency among historical analogues.

---

## Silver Analysis

* **Technical Structure:** Composite factor score {s_score:.1f}/100.
* **Gold/Silver Ratio:** {ratio_val:.2f} (Z-Score: {relative_value.get("latest_zscore_60d", 0.0):.2f})
* **Historical Analogues:** Matched {silver_analogues.get("analogue_count", 0)} historical market conditions.
* **5-Day Forward Probability:** {silver_analysis.get("signals", {}).get("5d", {}).get("positive_return_probability_pct", 50.0)}% positive return frequency among historical analogues.

---

## Macro Environment

* **Macro Regime:** {macro_reg}
* **DXY 20D Return:** {macro_regime.get("metrics", {}).get("dxy_20d_return", 0.0)*100:+.2f}%
* **US 10Y Yield 20D Change:** {macro_regime.get("metrics", {}).get("us10y_20d_change", 0.0):+.2f}%
* **VIX Level:** {macro_regime.get("metrics", {}).get("vix_level", 15.0):.1f}

---

## Historical Analogue Analysis

Top matched historical dates for Gold:
"""
        for top_a in gold_analogues.get("top_analogues", [])[:5]:
            md += f"- **{top_a['date']}** (Similarity Score: {top_a['similarity_score']}/100, Distance: {top_a['distance']:.2f})\n"

        md += f"""
---

## Forecast Research

*Note: The following statistics represent historical conditional probabilities under matching market conditions, NOT guaranteed future price outcomes.*

| Horizon | Gold Prob Pos | Gold Mean Ret | Silver Prob Pos | Silver Mean Ret |
|:-------:|:-------------:|:-------------:|:---------------:|:---------------:|
"""
        g_sigs = gold_analysis.get("signals", {})
        s_sigs = silver_analysis.get("signals", {})
        for h in ["1d", "3d", "5d", "10d", "20d", "60d"]:
            g_s = g_sigs.get(h, {})
            s_s = s_sigs.get(h, {})
            md += f"| {h.upper()} | {g_s.get('positive_return_probability_pct', 50)}% | {g_s.get('expected_mean_return_pct', 0.0):+.2f}% | {s_s.get('positive_return_probability_pct', 50)}% | {s_s.get('expected_mean_return_pct', 0.0):+.2f}% |\n"

        md += f"""
---

## Active Market Alerts ({len(alerts)})

"""
        if alerts:
            for alt in alerts:
                md += f"- **[{alt.get('severity', 'info').upper()}]** {alt.get('message')}\n"
        else:
            md += "- No active alert thresholds triggered.\n"

        md += f"""
---

## Data Quality Summary

* **Date Processed:** {quality_report.get('date')}
* **Valid Records:** {quality_report.get('valid_count', 0)}
* **Suspicious Records:** {quality_report.get('suspicious_count', 0)}
* **Warnings/Errors:** {len(quality_report.get('warnings', []))} warnings, {len(quality_report.get('errors', []))} errors.

---
*Generated automatically by Gold & Silver Intelligence Pipeline.*
"""
        return md
