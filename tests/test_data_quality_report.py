"""
Tests for data_engineering/data_quality_report.py

Validates:
  - Audit correctly identifies missing fields
  - Report includes all required sections
  - Anomaly detection catches known bad data
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data_engineering.data_quality_report import audit_dataframe, generate_report

TEST_CONFIG = {
    "sector": "automotive",
    "logging": {"level": "WARNING", "format": "%(message)s"},
}


def _make_complete_df():
    """Fully populated synthetic dataframe."""
    return pd.DataFrame([
        {
            "ticker": "HEALTHY", "name": "Healthy Co", "year": 2023,
            "total_assets": 1_000_000,
            "current_assets": 500_000,
            "current_liabilities": 250_000,
            "total_liabilities": 600_000,
            "retained_earnings": 300_000,
            "ebit": 100_000,
            "market_cap": 900_000,
            "revenue": 1_200_000,
            "interest_expense": 20_000,
            "accounts_payable": 100_000,
            "accounts_receivable": 150_000,
            "cogs": 700_000,
            "long_term_debt": 350_000,
            "stockholders_equity": 400_000,
            "z_score": 3.5, "credit_zone": "safe",
            "current_ratio": 2.0,
        },
        {
            "ticker": "STRESSED", "name": "Stressed Co", "year": 2023,
            "total_assets": 500_000,
            "current_assets": 100_000,
            "current_liabilities": 300_000,
            "total_liabilities": 450_000,
            "retained_earnings": -200_000,
            "ebit": -30_000,
            "market_cap": 50_000,
            "revenue": 300_000,
            "interest_expense": 40_000,
            "accounts_payable": 200_000,
            "accounts_receivable": 80_000,
            "cogs": 250_000,
            "long_term_debt": 150_000,
            "stockholders_equity": 50_000,
            "z_score": 0.8, "credit_zone": "distress",
            "current_ratio": 0.33,
        },
    ])


def _make_partial_df():
    """Dataframe with significant missing data."""
    df = _make_complete_df().copy()
    df.loc[0, "ebit"] = float("nan")
    df.loc[0, "retained_earnings"] = float("nan")
    df.loc[0, "market_cap"] = float("nan")
    df.loc[1, "revenue"] = float("nan")
    df.loc[1, "total_assets"] = float("nan")
    return df


class TestAuditDataframe:

    def test_missing_fields_detected(self):
        df = _make_partial_df()
        audit = audit_dataframe(df)
        missing = audit["missing_by_field"]
        # ebit, retained_earnings, market_cap should each show 50% missing
        assert missing.get("ebit", 0) == pytest.approx(0.5, rel=0.01)

    def test_no_missing_in_complete_df(self):
        df = _make_complete_df()
        audit = audit_dataframe(df)
        missing = audit["missing_by_field"]
        # All fields present → missing rate should be 0
        for field in ["total_assets", "revenue", "ebit"]:
            if field in missing:
                assert missing[field] == pytest.approx(0.0)

    def test_negative_assets_flagged(self):
        df = _make_complete_df()
        df.loc[0, "total_assets"] = -1_000  # anomaly
        audit = audit_dataframe(df)
        anomalies = audit["anomalies"]
        neg_asset_flags = [a for a in anomalies if "Negative total_assets" in a["issue"]]
        assert len(neg_asset_flags) >= 1

    def test_zero_revenue_flagged(self):
        df = _make_complete_df()
        df.loc[0, "revenue"] = 0
        audit = audit_dataframe(df)
        anomalies = audit["anomalies"]
        rev_flags = [a for a in anomalies if "revenue" in a["issue"].lower()]
        assert len(rev_flags) >= 1

    def test_z_score_coverage_computed(self):
        df = _make_complete_df()
        audit = audit_dataframe(df)
        zq = audit.get("z_score_quality", {})
        assert zq.get("computed", 0) == 2  # 2 rows, both have z_score
        assert zq.get("missing", 0) == 0

    def test_missing_z_score_counted(self):
        df = _make_complete_df()
        df.loc[0, "z_score"] = float("nan")
        audit = audit_dataframe(df)
        zq = audit["z_score_quality"]
        assert zq["missing"] == 1
        assert zq["pct_missing"] == pytest.approx(50.0, rel=0.01)


class TestGenerateReport:

    def test_report_contains_required_sections(self, tmp_path):
        df = _make_complete_df()
        output = tmp_path / "test_report.md"
        report = generate_report(df, TEST_CONFIG, str(output))
        assert "# Data Quality Report" in report
        assert "Field Completeness" in report
        assert "Firm-Level Completeness" in report
        assert "Anomaly Flags" in report
        assert "Z-Score Coverage" in report
        assert "Methodology Notes" in report

    def test_report_written_to_disk(self, tmp_path):
        df = _make_complete_df()
        output = tmp_path / "test_report.md"
        generate_report(df, TEST_CONFIG, str(output))
        assert output.exists()
        content = output.read_text()
        assert len(content) > 100

    def test_report_flags_stressed_firm(self, tmp_path):
        df = _make_complete_df()
        output = tmp_path / "test_report.md"
        report = generate_report(df, TEST_CONFIG, str(output))
        # STRESSED should appear in the report
        assert "STRESSED" in report

    def test_report_flags_high_missing_firm(self, tmp_path):
        df = _make_partial_df()
        # Create a firm with >50% missing
        worst = {col: float("nan") for col in df.columns}
        worst.update({"ticker": "WORST", "name": "Worst Co", "year": 2023,
                      "z_score": float("nan"), "credit_zone": "unknown"})
        df = pd.concat([df, pd.DataFrame([worst])], ignore_index=True)
        output = tmp_path / "test_report.md"
        report = generate_report(df, TEST_CONFIG, str(output))
        assert "WORST" in report
