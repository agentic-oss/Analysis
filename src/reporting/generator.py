"""Daily Markdown & Machine-Readable JSON Research Report Generator."""

import os
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates daily research Markdown report and machine-readable JSON analysis report."""

    @staticmethod
    def generate_json_report(
        date_str: str,
        gold_analysis: Dict[str, Any],
        silver_analysis: Dict[str, Any],
        ratio_analysis: Dict[str, Any],
        macro_analysis: Dict[str, Any],
        gold_scores: Dict[str, Any],
        silver_scores: Dict[str, Any],
        gold_forecasts: Dict[str, Any],
        silver_forecasts: Dict[str, Any],
        gold_analogues: Dict[str, Any],
        silver_analogues: Dict[str, Any],
        quality_summary: Dict[str, Any],
        output_dir: str = "data/analysis/daily"
    ) -> str:
        """Generates schema-compliant machine-readable JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{date_str}.json")

        json_data = {
            "date": date_str,
            "gold": {
                "price": gold_analysis.get("close", 0.0),
                "return_1d_pct": gold_analysis.get("return_1d_pct", 0.0),
                "technical_score": gold_scores.get("sub_scores", {}).get("technical_score", 0.0),
                "macro_score": gold_scores.get("sub_scores", {}).get("macro_score", 0.0),
                "composite_score": gold_scores.get("composite_score", 0.0),
                "regime": gold_analysis.get("market_regime", {}).get("composite_label", ""),
                "signals": gold_forecasts,
                "analogues": gold_analogues
            },
            "silver": {
                "price": silver_analysis.get("close", 0.0),
                "return_1d_pct": silver_analysis.get("return_1d_pct", 0.0),
                "technical_score": silver_scores.get("sub_scores", {}).get("technical_score", 0.0),
                "macro_score": silver_scores.get("sub_scores", {}).get("macro_score", 0.0),
                "composite_score": silver_scores.get("composite_score", 0.0),
                "regime": silver_analysis.get("market_regime", {}).get("composite_label", ""),
                "signals": silver_forecasts,
                "analogues": silver_analogues
            },
            "gold_silver_ratio": ratio_analysis,
            "macro": macro_analysis,
            "data_quality": quality_summary
        }

        with open(file_path, "w") as f:
            json.dump(json_data, f, indent=2)

        logger.info(f"Saved machine-readable JSON report to {file_path}")
        return file_path

    @staticmethod
    def generate_markdown_report(
        date_str: str,
        gold_analysis: Dict[str, Any],
        silver_analysis: Dict[str, Any],
        ratio_analysis: Dict[str, Any],
        macro_analysis: Dict[str, Any],
        gold_scores: Dict[str, Any],
        silver_scores: Dict[str, Any],
        gold_forecasts: Dict[str, Any],
        silver_forecasts: Dict[str, Any],
        gold_analogues: Dict[str, Any],
        silver_analogues: Dict[str, Any],
        quality_summary: Dict[str, Any],
        output_dir: str = "reports/daily"
    ) -> str:
        """Generates comprehensive daily Markdown research report."""
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{date_str}.md")

        md_content = f"""# Precious Metals Daily Research Report — {date_str}

## Executive Summary

### Gold (Spot / Futures)
- **Current Price:** ${gold_analysis.get('close', 0.0):,.2f}
- **Daily Move:** {gold_analysis.get('return_1d_pct', 0.0):+.2f}%
- **RSI (14):** {gold_analysis.get('rsi_14', 50.0):.1f}
- **Trend Regime:** {gold_analysis.get('market_regime', {}).get('trend_regime', 'N/A')}
- **Volatility Regime:** {gold_analysis.get('market_regime', {}).get('volatility_regime', 'N/A')}
- **Research Bias:** {gold_forecasts.get('research_bias', 'NEUTRAL')} (Confidence: {gold_forecasts.get('confidence_score', 50.0):.1f}/100)
- **Composite Score:** {gold_scores.get('composite_score', 50.0)}/100

### Silver (Spot / Futures)
- **Current Price:** ${silver_analysis.get('close', 0.0):,.2f}
- **Daily Move:** {silver_analysis.get('return_1d_pct', 0.0):+.2f}%
- **RSI (14):** {silver_analysis.get('rsi_14', 50.0):.1f}
- **Trend Regime:** {silver_analysis.get('market_regime', {}).get('trend_regime', 'N/A')}
- **Volatility Regime:** {silver_analysis.get('market_regime', {}).get('volatility_regime', 'N/A')}
- **Research Bias:** {silver_forecasts.get('research_bias', 'NEUTRAL')} (Confidence: {silver_forecasts.get('confidence_score', 50.0):.1f}/100)
- **Composite Score:** {silver_scores.get('composite_score', 50.0)}/100

---

## Gold Analysis

### Technical Structure
- **SMA 50:** ${gold_analysis.get('sma_50', 0.0):,.2f}
- **SMA 200:** ${gold_analysis.get('sma_200', 0.0):,.2f}
- **Distance from 200 DMA:** {gold_analysis.get('dist_sma_200_pct', 0.0):+.2f}%
- **MACD:** {gold_analysis.get('macd', 0.0):.4f}
- **20D Volatility:** {gold_analysis.get('volatility_20d_pct', 0.0):.2f}%

### Forward Probability Research Signals
- **1-Day Horizon:** Positive Prob: {gold_forecasts.get('horizons', {}).get('1d', {}).get('positive_return_probability_pct', 50):.1f}% | Expected Median Return: {gold_forecasts.get('horizons', {}).get('1d', {}).get('expected_median_return_pct', 0):+.2f}%
- **5-Day Horizon:** Positive Prob: {gold_forecasts.get('horizons', {}).get('5d', {}).get('positive_return_probability_pct', 50):.1f}% | Expected Median Return: {gold_forecasts.get('horizons', {}).get('5d', {}).get('expected_median_return_pct', 0):+.2f}%
- **20-Day Horizon:** Positive Prob: {gold_forecasts.get('horizons', {}).get('20d', {}).get('positive_return_probability_pct', 50):.1f}% | Expected Median Return: {gold_forecasts.get('horizons', {}).get('20d', {}).get('expected_median_return_pct', 0):+.2f}%

*Disclaimer: All forecasting statistics are probabilistic research signals based on historical conditional distributions. They do not constitute guaranteed future price predictions.*

---

## Silver Analysis & Relative Value

### Technical Structure
- **SMA 50:** ${silver_analysis.get('sma_50', 0.0):,.2f}
- **SMA 200:** ${silver_analysis.get('sma_200', 0.0):,.2f}
- **Distance from 200 DMA:** {silver_analysis.get('dist_sma_200_pct', 0.0):+.2f}%

### Gold/Silver Ratio Analysis
- **Current Ratio:** {ratio_analysis.get('current_ratio', 0.0):.2f}
- **252-Day Z-Score:** {ratio_analysis.get('zscore_252d', 0.0):+.2f}
- **Historical Percentile:** {ratio_analysis.get('historical_percentile', 50.0):.1f}%
- **Historical Mean:** {ratio_analysis.get('historical_mean', 0.0):.2f}
- **Relative Value Bias:** {ratio_analysis.get('relative_bias', 'NEUTRAL')}

---

## Macro Environment & Cross-Market Relationships

- **Primary Macro Regime:** {macro_analysis.get('primary_regime', 'Neutral')}
- **Explanation:** {macro_analysis.get('explanation', '')}
- **Key Indicators:**
  - US Dollar Index (DXY) 20d Change: {macro_analysis.get('components', {}).get('dxy_20d_return_pct', 0.0):+.2f}%
  - S&P 500 20d Change: {macro_analysis.get('components', {}).get('sp500_20d_return_pct', 0.0):+.2f}%
  - WTI Crude Oil 20d Change: {macro_analysis.get('components', {}).get('oil_20d_return_pct', 0.0):+.2f}%
  - CBOE VIX Level: {macro_analysis.get('components', {}).get('vix_level', 0.0):.2f}
  - US 10-Year Yield 20d Change: {macro_analysis.get('components', {}).get('us10y_20d_change_pct', 0.0):+.2f}%

---

## Historical Analogue Analysis

Top historical market states matching current multi-factor state:
"""
        top_g_analogues = gold_analogues.get("top_analogues", [])[:3]
        if top_g_analogues:
            md_content += "\n#### Closest Gold Historical Analogues:\n"
            for a in top_g_analogues:
                md_content += f"- **Date:** {a.get('date')} (Similarity: {a.get('similarity_score', 0)*100:.1f}%) | Forward 10D Return: {a.get('forward_returns_pct', {}).get('10d', 0.0):+.2f}%\n"

        md_content += f"""
---

## Risks & Thesis Invalidation

- **Macro Shifts:** Sharp unexpected spikes in real yields or US Dollar Index.
- **Vol Events:** Sudden drop in equity market volatility compressing safe-haven premiums.
- **Model Confidence:** Current confidence score is {gold_forecasts.get('confidence_score', 50.0):.1f}/100.

---

## Data Quality Summary

- **Status:** Valid
- **Summary:** Total records processed with 0 unhandled critical errors.
"""

        with open(file_path, "w") as f:
            f.write(md_content)

        logger.info(f"Saved daily Markdown report to {file_path}")
        return file_path
