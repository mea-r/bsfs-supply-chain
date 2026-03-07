"""
data_quality_report.py — Audit and report on data completeness.

Produces: data_quality_report.md

For each firm-year, flags:
  - Missing financial fields (NaN)
  - Z-score computed with imputed components
  - Companies with >50% missing fields (unreliable)
  - Unusual values (negative assets, zero revenue, etc.)
"""

import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

from utils.config import load_config

logger = logging.getLogger(__name__)

# Columns required for Z-score; all others are supplemental
CRITICAL_FIELDS = [
    "total_assets",
    "current_assets",
    "current_liabilities",
    "retained_earnings",
    "ebit",
    "market_cap",
    "revenue",
    "total_liabilities",
]
SUPPLEMENTAL_FIELDS = [
    "interest_expense",
    "accounts_payable",
    "accounts_receivable",
    "cogs",
    "long_term_debt",
    "stockholders_equity",
]


def audit_dataframe(df: pd.DataFrame) -> dict:
    """
    Perform comprehensive data quality audit on a financial dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Combined financial data with ratio columns.

    Returns
    -------
    dict
        Audit results with keys: missing_by_field, missing_by_firm,
        anomalies, z_score_quality.
    """
    results = {}

    # 1. Missingness by field
    all_data_cols = CRITICAL_FIELDS + SUPPLEMENTAL_FIELDS
    existing_cols = [c for c in all_data_cols if c in df.columns]
    missing_rates = df[existing_cols].isna().mean().sort_values(ascending=False)
    results["missing_by_field"] = missing_rates

    # 2. Missingness by firm
    missing_by_firm = (
        df.groupby("ticker")[existing_cols]
        .apply(lambda g: g.isna().mean().mean())
        .sort_values(ascending=False)
    )
    results["missing_by_firm"] = missing_by_firm

    # 3. Anomalies
    anomalies = []
    if "total_assets" in df.columns:
        neg_assets = df[df["total_assets"] < 0]
        if not neg_assets.empty:
            for _, row in neg_assets.iterrows():
                anomalies.append(
                    {
                        "ticker": row["ticker"],
                        "year": row["year"],
                        "issue": "Negative total_assets",
                        "value": row["total_assets"],
                    }
                )
    if "revenue" in df.columns:
        zero_rev = df[(df["revenue"] == 0) | (df["revenue"].isna())]
        for _, row in zero_rev.iterrows():
            anomalies.append(
                {
                    "ticker": row["ticker"],
                    "year": row["year"],
                    "issue": "Zero or missing revenue",
                    "value": row.get("revenue", float("nan")),
                }
            )
    if "current_ratio" in df.columns:
        extreme_cr = df[df["current_ratio"] > 20]
        for _, row in extreme_cr.iterrows():
            anomalies.append(
                {
                    "ticker": row["ticker"],
                    "year": row["year"],
                    "issue": "Extreme current ratio (>20)",
                    "value": row["current_ratio"],
                }
            )
    results["anomalies"] = anomalies

    # 4. Z-Score quality
    if "z_score" in df.columns:
        z_missing = df["z_score"].isna().sum()
        z_total = len(df)
        z_dist = df.groupby("credit_zone").size() if "credit_zone" in df.columns else {}
        results["z_score_quality"] = {
            "computed": z_total - z_missing,
            "missing": z_missing,
            "pct_missing": round(z_missing / z_total * 100, 1),
            "zone_distribution": z_dist.to_dict()
            if hasattr(z_dist, "to_dict")
            else z_dist,
        }

    return results


def generate_report(
    df: pd.DataFrame, config: dict, output_path: str = "data_quality_report.md"
) -> str:
    """
    Generate a Markdown data quality report.

    Parameters
    ----------
    df : pd.DataFrame
        Financial data with computed ratios.
    config : dict
        Loaded config.yaml.
    output_path : str
        Where to write the report.

    Returns
    -------
    str
        Report text (also written to disk).
    """
    audit = audit_dataframe(df)
    lines = []
    lines.append("# Data Quality Report")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\nSector: **{config.get('sector', 'N/A')}**")
    lines.append(
        f"\nFirms: {df['ticker'].nunique()}  |  "
        f"Firm-years: {len(df)}  |  "
        f"Years: {sorted(df['year'].unique().tolist())}"
    )

    lines.append("\n---\n")
    lines.append("## 1. Field Completeness\n")
    lines.append("| Field | Missing % |")
    lines.append("|-------|-----------|")
    for field, rate in audit["missing_by_field"].items():
        flag = " ⚠️" if rate > 0.3 else ""
        lines.append(f"| {field} | {rate * 100:.1f}%{flag} |")

    lines.append("\n---\n")
    lines.append("## 2. Firm-Level Completeness\n")
    lines.append("| Ticker | Missing % | Status |")
    lines.append("|--------|-----------|--------|")
    for ticker, rate in audit["missing_by_firm"].items():
        if rate > 0.5:
            status = "🔴 UNRELIABLE — >50% data missing"
        elif rate > 0.25:
            status = "🟡 CAUTION — elevated missingness"
        else:
            status = "🟢 OK"
        lines.append(f"| {ticker} | {rate * 100:.1f}% | {status} |")

    lines.append("\n---\n")
    lines.append("## 3. Anomaly Flags\n")
    anomalies = audit.get("anomalies", [])
    if anomalies:
        lines.append("| Ticker | Year | Issue | Value |")
        lines.append("|--------|------|-------|-------|")
        for a in anomalies:
            lines.append(
                f"| {a['ticker']} | {a['year']} | {a['issue']} | {a['value']} |"
            )
    else:
        lines.append("No anomalies detected.")

    lines.append("\n---\n")
    lines.append("## 4. Z-Score Coverage\n")
    zq = audit.get("z_score_quality", {})
    if zq:
        lines.append(
            f"- Computed: **{zq['computed']}** of {zq['computed'] + zq['missing']} firm-years"
        )
        lines.append(f"- Missing: **{zq['pct_missing']}%**")
        lines.append(f"\nZone distribution: {zq.get('zone_distribution', {})}")

    lines.append("\n---\n")
    lines.append("## 5. Methodology Notes\n")
    lines.append(
        "- All financial data sourced from SEC EDGAR XBRL API (10-K annual filings)."
    )
    lines.append(
        "- Market cap sourced from yfinance (year-end close × shares outstanding)."
    )
    lines.append(
        "- Supply chain edges sourced from documented 10-K disclosures; "
        "inferred edges tagged `inferred_sector_structure`."
    )
    lines.append(
        "- Missing values are **never silently imputed**. "
        "Z-scores with >2 missing components return NaN."
    )
    lines.append("- All transformations are code-reproducible (no manual steps).")

    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)
    logger.info(f"Data quality report saved → {output_path}")
    return report


def run(config_path: str = "config.yaml") -> str:
    """
    Entry point: load scored data and generate quality report.

    Parameters
    ----------
    config_path : str
        Path to config.yaml.

    Returns
    -------
    str
        Report text.
    """
    config = load_config(config_path)
    logging.basicConfig(
        level=config["logging"]["level"],
        format=config["logging"]["format"],
    )
    scores_path = Path("risk_framework") / "scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(
            f"scores.csv not found at {scores_path}. Run ratio_calculator.py first."
        )
    df = pd.read_csv(scores_path)
    return generate_report(df, config)


if __name__ == "__main__":
    run()
