"""
Tests for data_engineering/ratio_calculator.py

Economic ground-truth checks:
  - Z-Score formula correctness against Altman (1968)
  - Credit zone boundaries match Altman's empirical thresholds
  - Ratio calculations match standard definitions
  - Missing data handling (NaN propagation, not silent imputation)
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data_engineering.ratio_calculator import (
    compute_altman_z_score,
    classify_credit_zone,
    compute_ratios,
)

# Minimal config matching config.yaml structure for tests
TEST_CONFIG = {
    "ratios": {
        "z_score": {"w1": 1.2, "w2": 1.4, "w3": 3.3, "w4": 0.6, "w5": 1.0},
        "credit_zones": {"safe_threshold": 2.99, "grey_threshold": 1.81},
        "liquidity": {"current_ratio_stress": 1.0},
    }
}
Z_WEIGHTS = TEST_CONFIG["ratios"]["z_score"]
ZONES = TEST_CONFIG["ratios"]["credit_zones"]


# ------------------------------------------------------------------
# Z-Score computation tests
# ------------------------------------------------------------------

class TestAltmanZScore:

    def _make_row(self, **kwargs):
        """Helper: create a pd.Series with financial data."""
        defaults = {
            "total_assets": 1_000_000,
            "current_assets": 500_000,
            "current_liabilities": 300_000,
            "retained_earnings": 200_000,
            "ebit": 100_000,
            "market_cap": 800_000,
            "total_liabilities": 600_000,
            "revenue": 1_200_000,
        }
        defaults.update(kwargs)
        return pd.Series(defaults)

    def test_formula_correctness(self):
        """Z-score must match manual calculation of Altman formula."""
        row = self._make_row()
        ta = 1_000_000
        x1 = (500_000 - 300_000) / ta   # 0.2
        x2 = 200_000 / ta               # 0.2
        x3 = 100_000 / ta               # 0.1
        x4 = 800_000 / 600_000          # 1.333...
        x5 = 1_200_000 / ta             # 1.2

        expected = (1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5)
        result = compute_altman_z_score(row, Z_WEIGHTS)
        assert result == pytest.approx(expected, rel=1e-3)

    def test_safe_zone_firm(self):
        """A financially healthy firm should have Z > 2.99 (safe zone)."""
        # Simulate a healthy, well-capitalized firm
        row = self._make_row(
            current_assets=800_000,    # high liquidity
            current_liabilities=200_000,
            retained_earnings=500_000, # strong accumulated profitability
            ebit=300_000,              # high operating margin
            market_cap=3_000_000,      # market well above liabilities
            total_liabilities=400_000,
            revenue=2_000_000,         # high revenue relative to assets
        )
        z = compute_altman_z_score(row, Z_WEIGHTS)
        assert z > 2.99, f"Healthy firm should be in safe zone, got Z={z:.2f}"

    def test_distress_zone_firm(self):
        """A financially stressed firm should have Z ≤ 1.81 (distress zone)."""
        row = self._make_row(
            current_assets=100_000,    # low liquidity
            current_liabilities=600_000,  # current liabilities exceed current assets!
            retained_earnings=-400_000,   # accumulated losses
            ebit=-50_000,              # operating losses
            market_cap=100_000,        # market has little confidence
            total_liabilities=900_000,
            revenue=300_000,           # low revenue relative to assets
        )
        z = compute_altman_z_score(row, Z_WEIGHTS)
        assert z <= 1.81, f"Distressed firm should be in distress zone, got Z={z:.2f}"

    def test_missing_total_assets_returns_nan(self):
        """If total_assets is missing, Z-score must be NaN (not imputed)."""
        row = self._make_row(total_assets=float("nan"))
        z = compute_altman_z_score(row, Z_WEIGHTS)
        assert pd.isna(z), "Z-score should be NaN when total_assets is missing"

    def test_zero_total_assets_returns_nan(self):
        """Z-score is undefined when total_assets = 0."""
        row = self._make_row(total_assets=0)
        z = compute_altman_z_score(row, Z_WEIGHTS)
        assert pd.isna(z)

    def test_all_components_nan_returns_nan(self):
        """When all components are NaN, Z-score must be NaN."""
        row = pd.Series({
            "total_assets": 1_000_000,
            "current_assets": float("nan"),
            "current_liabilities": float("nan"),
            "retained_earnings": float("nan"),
            "ebit": float("nan"),
            "market_cap": float("nan"),
            "total_liabilities": float("nan"),
            "revenue": float("nan"),
        })
        z = compute_altman_z_score(row, Z_WEIGHTS)
        assert pd.isna(z), "Z-score with no valid components should be NaN"

    def test_components_directionally_correct(self):
        """Higher retained earnings should increase Z-score (all else equal)."""
        base = self._make_row()
        high_re = self._make_row(retained_earnings=800_000)
        z_base = compute_altman_z_score(base, Z_WEIGHTS)
        z_high = compute_altman_z_score(high_re, Z_WEIGHTS)
        assert z_high > z_base, "Higher retained earnings must increase Z-score"

    def test_interest_rate_shock_reduces_z(self):
        """
        Economic validation: increasing interest expense → lower EBIT → lower Z-score.
        Mimics S1 shock (interest rate spike).
        """
        row_pre = self._make_row(ebit=200_000)
        row_post = self._make_row(ebit=100_000)  # half EBIT after rate spike
        z_pre = compute_altman_z_score(row_pre, Z_WEIGHTS)
        z_post = compute_altman_z_score(row_post, Z_WEIGHTS)
        assert z_post < z_pre, "Z-score must fall when EBIT falls (interest rate shock)"


# ------------------------------------------------------------------
# Credit Zone classification tests
# ------------------------------------------------------------------

class TestCreditZoneClassification:

    def test_safe_zone(self):
        assert classify_credit_zone(3.5, 2.99, 1.81) == "safe"
        assert classify_credit_zone(2.991, 2.99, 1.81) == "safe"

    def test_grey_zone(self):
        assert classify_credit_zone(2.5, 2.99, 1.81) == "grey"
        assert classify_credit_zone(1.82, 2.99, 1.81) == "grey"
        assert classify_credit_zone(2.99, 2.99, 1.81) == "grey"  # exactly 2.99 → not > 2.99

    def test_distress_zone(self):
        assert classify_credit_zone(1.0, 2.99, 1.81) == "distress"
        assert classify_credit_zone(1.81, 2.99, 1.81) == "distress"
        assert classify_credit_zone(-5.0, 2.99, 1.81) == "distress"

    def test_nan_returns_unknown(self):
        assert classify_credit_zone(float("nan"), 2.99, 1.81) == "unknown"


# ------------------------------------------------------------------
# Ratio computation tests
# ------------------------------------------------------------------

class TestComputeRatios:

    def _make_df(self):
        """Create a minimal synthetic dataframe for ratio testing."""
        return pd.DataFrame([
            {
                "ticker": "TEST",
                "name": "Test Co",
                "year": 2023,
                "total_assets": 1_000_000,
                "current_assets": 500_000,
                "current_liabilities": 250_000,
                "retained_earnings": 300_000,
                "ebit": 120_000,
                "market_cap": 900_000,
                "total_liabilities": 600_000,
                "revenue": 1_500_000,
                "interest_expense": 20_000,
                "accounts_payable": 100_000,
                "accounts_receivable": 150_000,
                "cogs": 900_000,
                "long_term_debt": 350_000,
                "stockholders_equity": 400_000,
            }
        ])

    def test_current_ratio(self):
        df = self._make_df()
        result = compute_ratios(df, TEST_CONFIG)
        expected = 500_000 / 250_000  # = 2.0
        assert result["current_ratio"].iloc[0] == pytest.approx(expected)

    def test_interest_coverage_ratio(self):
        df = self._make_df()
        result = compute_ratios(df, TEST_CONFIG)
        expected = 120_000 / 20_000  # = 6.0
        assert result["interest_coverage_ratio"].iloc[0] == pytest.approx(expected)

    def test_days_payable_outstanding(self):
        df = self._make_df()
        result = compute_ratios(df, TEST_CONFIG)
        expected = (100_000 / 900_000) * 365  # ≈ 40.6 days
        assert result["days_payable_outstanding"].iloc[0] == pytest.approx(expected, rel=1e-3)

    def test_days_sales_outstanding(self):
        df = self._make_df()
        result = compute_ratios(df, TEST_CONFIG)
        expected = (150_000 / 1_500_000) * 365  # = 36.5 days
        assert result["days_sales_outstanding"].iloc[0] == pytest.approx(expected, rel=1e-3)

    def test_debt_to_equity(self):
        df = self._make_df()
        result = compute_ratios(df, TEST_CONFIG)
        expected = 600_000 / 400_000  # = 1.5
        assert result["debt_to_equity"].iloc[0] == pytest.approx(expected)

    def test_zero_denominator_handled(self):
        """Division by zero must produce NaN, not an error."""
        df = self._make_df()
        df["current_liabilities"] = 0
        result = compute_ratios(df, TEST_CONFIG)
        assert pd.isna(result["current_ratio"].iloc[0])

    def test_credit_zone_column_created(self):
        df = self._make_df()
        result = compute_ratios(df, TEST_CONFIG)
        assert "credit_zone" in result.columns
        assert result["credit_zone"].iloc[0] in ("safe", "grey", "distress", "unknown")

    def test_working_capital_correct(self):
        df = self._make_df()
        result = compute_ratios(df, TEST_CONFIG)
        expected = 500_000 - 250_000  # = 250_000
        assert result["working_capital"].iloc[0] == pytest.approx(expected)
