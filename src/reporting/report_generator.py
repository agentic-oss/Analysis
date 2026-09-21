import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class DailyReportGenerator:
    """Generates daily human-readable Markdown research reports and machine-readable JSON files."""

    @staticmethod
    def generate_markdown_report(
        date_str: str,
        gold_analysis: Dict[str, Any],
        silver_analysis: Dict[str, Any],
        ratio_analysis: Dict[str, Any],
        macro_regime: Dict[str, Any],
        gold_market_regime: Dict[str, Any],
        silver_market_regime: Dict[str, Any],
        analogues: List[Dict[str, Any]],
        analogue_stats: Dict[str, Any],
        forward_signals: Dict[str, Any],
        scores: Dict[str, Any],
        alerts: Dict[str, Any],
        data_quality: Dict[str, Any]
    ) -> str:

        md = []
        md.append(f"# Precious Metals Daily Research Report — {date_str}")
        md.append("")
        md.append("> **Research Disclaimer**: All price forecasts and statistical outputs are probabilistic research signals based on historical patterns and quantitative models. They do not constitute guaranteed future prices or financial advice.")
        md.append("")

        # Executive Summary
        md.append("## Executive Summary")
        md.append("")
        md.append("### Gold")
        if gold_analysis:
            g_inr = f" (₹{gold_analysis.get('price_inr_10g'):,.2f} / 10g)" if gold_analysis.get("price_inr_10g") else ""
            md.append(f"- **Current Price**: ${gold_analysis.get('price', 0):,.2f} USD/oz{g_inr}")
            md.append(f"- **Daily Move**: {gold_analysis.get('daily_move_pct', 0):+.2f}%")
            md.append(f"- **Weekly Move**: {gold_analysis.get('weekly_move_pct', 0):+.2f}%")
            md.append(f"- **Monthly Move**: {gold_analysis.get('monthly_move_pct', 0):+.2f}%")
            md.append(f"- **Trend**: {gold_analysis.get('trend', 'N/A')}")
            md.append(f"- **Volatility**: {gold_analysis.get('volatility', 'N/A')} ({gold_analysis.get('volatility_20d_ann', 0):.1f}% annualized)")
            md.append(f"- **Research Bias**: {gold_analysis.get('research_bias', 'N/A')}")
        else:
            md.append("- Data unavailable")
        md.append("")

        md.append("### Silver")
        if silver_analysis:
            s_inr = f" (₹{silver_analysis.get('price_inr_kg'):,.2f} / kg)" if silver_analysis.get("price_inr_kg") else ""
            md.append(f"- **Current Price**: ${silver_analysis.get('price', 0):,.2f} USD/oz{s_inr}")
            md.append(f"- **Daily Move**: {silver_analysis.get('daily_move_pct', 0):+.2f}%")
            md.append(f"- **Weekly Move**: {silver_analysis.get('weekly_move_pct', 0):+.2f}%")
            md.append(f"- **Monthly Move**: {silver_analysis.get('monthly_move_pct', 0):+.2f}%")
            md.append(f"- **Trend**: {silver_analysis.get('trend', 'N/A')}")
            md.append(f"- **Volatility**: {silver_analysis.get('volatility', 'N/A')} ({silver_analysis.get('volatility_20d_ann', 0):.1f}% annualized)")
            md.append(f"- **Research Bias**: {silver_analysis.get('research_bias', 'N/A')}")
        else:
            md.append("- Data unavailable")
        md.append("")

        # Gold Analysis
        md.append("## Gold Analysis")
        md.append("")
        if gold_analysis:
            ti = gold_analysis.get("technical_indicators", {})
            kl = gold_analysis.get("key_levels", {})
            md.append(f"- **Technical Structure**: RSI(14)={ti.get('rsi_14', 50):.1f}, MACD Hist={ti.get('macd_hist', 0):.2f}, Distance to 200 DMA={ti.get('dist_sma_200_pct', 0):+.1f}%")
            md.append(f"- **Key Support**: ${kl.get('key_support', 0):,.2f}")
            md.append(f"- **Key Resistance**: ${kl.get('key_resistance', 0):,.2f}")
            md.append(f"- **20-Day Range**: ${kl.get('low_20d', 0):,.2f} – ${kl.get('high_20d', 0):,.2f}")
        md.append("")

        # Silver Analysis
        md.append("## Silver Analysis")
        md.append("")
        if silver_analysis:
            ti = silver_analysis.get("technical_indicators", {})
            kl = silver_analysis.get("key_levels", {})
            md.append(f"- **Technical Structure**: RSI(14)={ti.get('rsi_14', 50):.1f}, MACD Hist={ti.get('macd_hist', 0):.2f}")
            md.append(f"- **Key Support**: ${kl.get('key_support', 0):,.2f}")
            md.append(f"- **Key Resistance**: ${kl.get('key_resistance', 0):,.2f}")
        if ratio_analysis:
            md.append(f"- **Gold/Silver Ratio**: {ratio_analysis.get('latest_ratio', 0):.2f} (252d Z-Score: {ratio_analysis.get('latest_zscore', 0):+.2f}, Percentile: {ratio_analysis.get('latest_percentile', 50):.1f}%)")
            md.append(f"- **Relative Value Regime**: {ratio_analysis.get('mean_reversion_signal', 'NEUTRAL')}")
        md.append("")

        # Macro Environment
        md.append("## Macro Environment")
        md.append("")
        if macro_regime:
            md.append(f"- **Primary Macro Regime**: {macro_regime.get('primary_regime', 'Neutral')}")
            md.append(f"- **Risk Sentiment**: {macro_regime.get('risk_sentiment', 'Neutral')}")
            md.append(f"- **Interest Rates Stance**: {macro_regime.get('rate_stance', 'Neutral')}")
            md.append(f"- **US Dollar Stance**: {macro_regime.get('dxy_stance', 'Neutral')}")
            md.append(f"- **Inflation Stance**: {macro_regime.get('inflation_stance', 'Neutral')}")
            md.append(f"- **Macro Context**: {macro_regime.get('explanation', '')}")
        md.append("")

        # Historical Analogue Analysis
        md.append("## Historical Analogue Analysis")
        md.append("")
        if analogues:
            md.append(f"Top {len(analogues)} most similar historical market states detected:")
            md.append("")
            md.append("| Date | Similarity Score | Historical Price | 5D Fwd Return | 20D Fwd Return |")
            md.append("| --- | --- | --- | --- | --- |")
            for a in analogues[:5]:
                out = a.get("forward_outcomes", {})
                ret_5 = f"{out.get('5d', 0):+.2f}%" if "5d" in out else "N/A"
                ret_20 = f"{out.get('20d', 0):+.2f}%" if "20d" in out else "N/A"
                md.append(f"| {a.get('date')} | {a.get('similarity_score')}/100 | ${a.get('historical_price'):,.2f} | {ret_5} | {ret_20} |")
        md.append("")

        # Forecast Research
        md.append("## Forecast Research (Probabilistic Signals)")
        md.append("")
        horizons_data = forward_signals.get("horizons", {}) if forward_signals else {}
        if horizons_data:
            md.append("| Horizon | Probability Positive | Expected Return | Median Return | Implied Price | Research Bias |")
            md.append("| --- | --- | --- | --- | --- | --- |")
            for h_key, h_info in horizons_data.items():
                md.append(f"| {h_key.upper()} | {h_info.get('probability_positive_pct')}% | {h_info.get('expected_return_pct'):+.2f}% | {h_info.get('median_return_pct'):+.2f}% | ${h_info.get('implied_median_target_price'):,.2f} | {h_info.get('signal_bias')} |")
        md.append("")

        # Composite Scoring
        md.append("## Multi-Factor Composite Scores")
        md.append("")
        if scores:
            md.append(f"- **Technical Score**: {scores.get('technical_score')}/100")
            md.append(f"- **Macro Score**: {scores.get('macro_score')}/100")
            md.append(f"- **Momentum Score**: {scores.get('momentum_score')}/100")
            md.append(f"- **Volatility Score**: {scores.get('volatility_score')}/100")
            md.append(f"- **Relative Value Score**: {scores.get('relative_value_score')}/100")
            md.append(f"- **Historical Pattern Score**: {scores.get('historical_pattern_score')}/100")
            md.append(f"- **COMPOSITE SCORE**: **{scores.get('composite_score')}/100**")
        md.append("")

        # Alerts & Risks
        md.append("## Alerts & Risk Factors")
        md.append("")
        if alerts and alerts.get("alerts"):
            for alt in alerts["alerts"]:
                md.append(f"- ⚠️ **[{alt.get('severity')}] {alt.get('type')}**: {alt.get('message')}")
        else:
            md.append("- No high-severity threshold alerts triggered today.")
        md.append("")

        # Data Quality
        md.append("## Data Quality Status")
        md.append("")
        if data_quality:
            md.append(f"- Total records ingested: {data_quality.get('total_records', 0)}")
            md.append(f"- Warnings: {len(data_quality.get('warnings', []))}")
            md.append(f"- Errors: {len(data_quality.get('errors', []))}")
        md.append("")

        report_str = "\n".join(md)
        return report_str

    @staticmethod
    def generate_json_report(
        date_str: str,
        gold_analysis: Dict[str, Any],
        silver_analysis: Dict[str, Any],
        ratio_analysis: Dict[str, Any],
        macro_regime: Dict[str, Any],
        gold_market_regime: Dict[str, Any],
        silver_market_regime: Dict[str, Any],
        analogues: List[Dict[str, Any]],
        analogue_stats: Dict[str, Any],
        forward_signals: Dict[str, Any],
        scores: Dict[str, Any],
        alerts: Dict[str, Any],
        data_quality: Dict[str, Any]
    ) -> Dict[str, Any]:

        return {
            "date": date_str,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "gold": gold_analysis,
            "silver": silver_analysis,
            "gold_silver_ratio": ratio_analysis,
            "macro_regime": macro_regime,
            "market_regime": {
                "gold": gold_market_regime,
                "silver": silver_market_regime
            },
            "historical_analogues": analogues,
            "analogue_statistics": analogue_stats,
            "forward_probabilistic_signals": forward_signals,
            "scores": scores,
            "alerts": alerts,
            "data_quality": data_quality
        }
