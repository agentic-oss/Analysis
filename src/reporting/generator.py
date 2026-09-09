"""
Daily Research Report Generator.
Generates comprehensive Markdown daily research reports at reports/daily/YYYY-MM-DD.md
and machine-readable JSON files at data/analysis/daily/YYYY-MM-DD.json.
Follows strict non-sensational, probabilistic research standards.
"""

from datetime import datetime
import json
import logging
import os
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class DailyReportGenerator:
    """Generates structured Markdown research reports and JSON analysis snapshots."""

    def generate_markdown_report(self, analysis_data: Dict[str, Any], output_filepath: str):
        """Creates professional Markdown report adhering to required structure."""
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

        date_str = analysis_data.get("date", datetime.now().strftime("%Y-%m-%d"))
        gold = analysis_data.get("gold", {})
        silver = analysis_data.get("silver", {})
        macro = analysis_data.get("macro", {})
        gs_ratio = analysis_data.get("gold_silver_ratio", {})
        quality = analysis_data.get("data_quality", {})
        alerts = analysis_data.get("alerts", [])

        md = []
        md.append(f"# Precious Metals Daily Research Report — {date_str}\n")

        md.append("## Executive Summary\n")

        # Gold Summary
        g_price = gold.get("price", 0.0)
        g_1d = gold.get("return_1d", 0.0)
        g_5d = gold.get("return_5d", 0.0)
        g_20d = gold.get("return_20d", 0.0)
        g_bias = gold.get("research_bias", "NEUTRAL")
        g_comp = gold.get("composite_score", 50.0)
        md.append(f"### Gold (Spot USD/oz)\n")
        md.append(f"- **Current Price:** ${g_price:,.2f}")
        md.append(f"- **Daily Move:** {g_1d:+.2f}% | **Weekly Move:** {g_5d:+.2f}% | **Monthly Move:** {g_20d:+.2f}%")
        md.append(f"- **Trend Regime:** {gold.get('trend_regime', 'NEUTRAL')} | **Volatility:** {gold.get('volatility_regime', 'NORMAL')}")
        md.append(f"- **Composite Research Score:** {g_comp:.1f}/100 | **Research Bias:** {g_bias}\n")

        # Silver Summary
        s_price = silver.get("price", 0.0)
        s_1d = silver.get("return_1d", 0.0)
        s_5d = silver.get("return_5d", 0.0)
        s_20d = silver.get("return_20d", 0.0)
        s_bias = silver.get("research_bias", "NEUTRAL")
        s_comp = silver.get("composite_score", 50.0)
        md.append(f"### Silver (Spot USD/oz)\n")
        md.append(f"- **Current Price:** ${s_price:,.2f}")
        md.append(f"- **Daily Move:** {s_1d:+.2f}% | **Weekly Move:** {s_5d:+.2f}% | **Monthly Move:** {s_20d:+.2f}%")
        md.append(f"- **Trend Regime:** {silver.get('trend_regime', 'NEUTRAL')} | **Volatility:** {silver.get('volatility_regime', 'NORMAL')}")
        md.append(f"- **Composite Research Score:** {s_comp:.1f}/100 | **Research Bias:** {s_bias}\n")

        # Active Alerts
        if alerts:
            md.append("### Active Market Alerts\n")
            for a in alerts:
                md.append(f"- **[{a.get('type')}]** ({a.get('severity')}) — {a.get('message')}")
            md.append("")

        # Gold Analysis
        md.append("## Gold Detailed Analysis\n")
        md.append(f"- **Technical Structure:** RSI: {gold.get('rsi_14', 50):.1f} | MACD: {gold.get('macd', 0):.2f} | ATR: ${gold.get('atr_14', 0):.2f}")
        md.append(f"- **Distance to 200 SMA:** {gold.get('dist_sma_200_pct', 0):+.2f}% | **Distance to 52W High:** {gold.get('dist_52w_high_pct', 0):+.2f}%")
        g_fwd = gold.get("forward_statistics", {})
        if g_fwd:
            md.append("- **Forward Probabilities (Historical Analogues & Models):**")
            for h in ["1d", "3d", "5d", "10d", "20d", "60d"]:
                if h in g_fwd:
                    st = g_fwd[h]
                    md.append(f"  - **{h.upper()} Horizon:** Prob Positive: {st.get('probability_positive', 50):.1f}% | Exp Return: {st.get('expected_return', 0):+.2f}%")
        md.append("")

        # Silver Analysis & Gold/Silver Ratio
        md.append("## Silver & Relative Value Analysis\n")
        md.append(f"- **Gold/Silver Ratio:** {gs_ratio.get('current_ratio', 0):.2f} (Historical Percentile: {gs_ratio.get('historical_percentile', 50):.1f}%, Z-Score: {gs_ratio.get('zscore', 0):+.2f})")
        md.append(f"- **Relative Value Signal:** {gs_ratio.get('signal', 'NEUTRAL')}")
        md.append(f"- **Technical Structure:** RSI: {silver.get('rsi_14', 50):.1f} | ATR: ${silver.get('atr_14', 0):.2f}\n")

        # Macro Environment
        md.append("## Macro Environment & Cross-Market Drivers\n")
        md.append(f"- **Macro Regime:** {macro.get('macro_regime', 'NEUTRAL')}")
        md.append(f"- **DXY Index:** {macro.get('dxy_close', 0):.2f} | **US 10Y Yield:** {macro.get('us10y_close', 0):.2f}%")
        md.append(f"- **Crude Oil (Brent):** ${macro.get('oil_brent_close', 0):.2f} | **Equity Volatility (VIX):** {macro.get('vix_close', 0):.2f}")
        md.append(f"- **Indian Market Indicators:** NIFTY 50: {macro.get('nifty_close', 0):,.0f} | USD/INR: ₹{macro.get('usdinr_close', 0):.2f}\n")

        # Historical Analogue Analysis
        md.append("## Historical Analogue Analysis\n")
        top_ana = gold.get("top_analogues", [])
        if top_ana:
            md.append("Closest historical market conditions observed:")
            for item in top_ana[:3]:
                md.append(f"- **Date:** {item.get('historical_date')} | **Similarity:** {item.get('similarity_score', 0):.1f}% | **RSI:** {item.get('rsi_14', 0):.1f}")
        md.append("")

        # Risks & Research Thesis Invalidation
        md.append("## Risks & Thesis Invalidation\n")
        md.append("- **Upside Risks to Bias:** Unexpected dovish monetary pivot, geopolitical escalation, sharp Dollar devaluation.")
        md.append("- **Downside Risks to Bias:** Aggressive rate hikes, real yield surges, sharp Liquidity sell-offs.\n")

        # Data Quality
        md.append("## Data Quality & System Information\n")
        md.append(f"- **Evaluated Records:** {quality.get('total_records', 0)} | **Valid:** {quality.get('valid_count', 0)} | **Suspicious:** {quality.get('suspicious_count', 0)} | **Invalid:** {quality.get('invalid_count', 0)}")
        if quality.get("warnings"):
            md.append(f"- **Warnings:** {', '.join(quality.get('warnings')[:3])}")
        md.append(f"- *Report generated automatically on {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}.*")

        with open(output_filepath, "w") as f:
            f.write("\n".join(md))

        logger.info(f"Markdown report generated at {output_filepath}")

    def generate_json_report(self, analysis_data: Dict[str, Any], output_filepath: str):
        """Generates machine-readable JSON report."""
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        with open(output_filepath, "w") as f:
            json.dump(analysis_data, f, indent=2, default=str)
        logger.info(f"JSON report written to {output_filepath}")
