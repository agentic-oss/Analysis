import os
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generates daily Markdown report, machine-readable JSON, and daily alerts."""

    @staticmethod
    def generate_daily_markdown_report(
        date_str: str,
        gold_analysis: Dict[str, Any],
        silver_analysis: Dict[str, Any],
        ratio_analysis: Dict[str, Any],
        macro_regime: Dict[str, Any],
        gold_forecasts: Dict[str, Any],
        silver_forecasts: Dict[str, Any],
        gold_analogues: Dict[str, Any],
        silver_analogues: Dict[str, Any],
        data_quality: Dict[str, Any]
    ) -> str:
        report = f"""# Precious Metals Daily Research Report — {date_str}

## Executive Summary

### Gold (USD/oz)
* **Current Price:** ${gold_analysis.get('current_price', 0.0):,.2f}
* **Daily Move:** {gold_analysis.get('daily_return_pct', 0.0):+.2f}%
* **Weekly Move:** {gold_analysis.get('weekly_return_pct', 0.0):+.2f}%
* **Monthly Move:** {gold_analysis.get('monthly_return_pct', 0.0):+.2f}%
* **RSI (14):** {gold_analysis.get('rsi_14', 50.0):.1f}
* **Research Bias:** {macro_regime.get('primary_regime', 'Neutral')} Macro Regime

### Silver (USD/oz)
* **Current Price:** ${silver_analysis.get('current_price', 0.0):,.2f}
* **Daily Move:** {silver_analysis.get('daily_return_pct', 0.0):+.2f}%
* **Weekly Move:** {silver_analysis.get('weekly_return_pct', 0.0):+.2f}%
* **Monthly Move:** {silver_analysis.get('monthly_return_pct', 0.0):+.2f}%
* **RSI (14):** {silver_analysis.get('rsi_14', 50.0):.1f}

---

## Gold Analysis
* **Distance from 200 SMA:** {gold_analysis.get('dist_sma_200_pct', 0.0):+.2f}%
* **Distance from 52-Week High:** {gold_analysis.get('dist_52w_high_pct', 0.0):+.2f}%
* **Current Drawdown:** {gold_analysis.get('drawdown_pct', 0.0):.2f}%
* **20-Day Volatility:** {gold_analysis.get('volatility_20d', 0.0):.1f}%

### Gold Forward Probability Research Signals
* **10-Day Horizon:** Positive Return Probability: {gold_forecasts.get('forecasts', {}).get('10d', {}).get('probability_positive_pct', 50.0)}% | Mean Return: {gold_forecasts.get('forecasts', {}).get('10d', {}).get('expected_mean_return_pct', 0.0):+.2f}%
* **20-Day Horizon:** Positive Return Probability: {gold_forecasts.get('forecasts', {}).get('20d', {}).get('probability_positive_pct', 50.0)}% | Mean Return: {gold_forecasts.get('forecasts', {}).get('20d', {}).get('expected_mean_return_pct', 0.0):+.2f}%

---

## Silver Analysis
* **Distance from 200 SMA:** {silver_analysis.get('dist_sma_200_pct', 0.0):+.2f}%
* **Gold/Silver Ratio:** {ratio_analysis.get('current_ratio', 0.0)} (Z-score: {ratio_analysis.get('z_score', 0.0)})
* **Historical Percentile:** {ratio_analysis.get('percentile', 50.0)}th percentile

### Silver Forward Probability Research Signals
* **10-Day Horizon:** Positive Return Probability: {silver_forecasts.get('forecasts', {}).get('10d', {}).get('probability_positive_pct', 50.0)}% | Mean Return: {silver_forecasts.get('forecasts', {}).get('10d', {}).get('expected_mean_return_pct', 0.0):+.2f}%

---

## Macro Environment
* **Primary Macro Regime:** {macro_regime.get('primary_regime', 'Neutral')}
* **Key Components:**
  * DXY 20D Return: {macro_regime.get('components', {}).get('dxy_return_20d', 0.0):+.2f}%
  * US 10Y Yield: {macro_regime.get('components', {}).get('us10y_yield', 0.0):.2f}%
  * VIX Index: {macro_regime.get('components', {}).get('vix', 0.0):.2f}

---

## Historical Analogue Matches
* **Gold Sample Count:** {gold_analogues.get('sample_count', 0)} matches found in historical database.
* **Silver Sample Count:** {silver_analogues.get('sample_count', 0)} matches found in historical database.

---

## Risks & Thesis Invalidation
* Sudden surge in real yields or US Dollar Index (DXY) momentum.
* Extreme market volatility expansion.

---

## Data Quality Summary
* Total Records Inspected: {data_quality.get('records_count', 0)}
* Warnings / Anomalies: {len(data_quality.get('warnings', []))}
* Errors / Invalid Records: {len(data_quality.get('errors', []))}

*Note: All probabilities and forecasts are conditional historical statistics for research purposes and do not represent guaranteed future market outcomes.*
"""
        return report.strip()
