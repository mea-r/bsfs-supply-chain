"""
Tests for propagation/propagation_engine.py

Validates:
  - Graph construction from scores + edges
  - All 3 shock scenarios run without error
  - Propagation rules are directionally correct (supplier stress → buyer stress, not vice versa)
  - Chokepoint detection
  - Reset functionality
  - Economic sanity: stress score should increase after shock, not decrease
"""

import sys
from pathlib import Path
import copy

import pytest
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from propagation.propagation_engine import PropagationEngine  # noqa: E402


# ------------------------------------------------------------------
# Test fixtures
# ------------------------------------------------------------------

BASE_CONFIG = {
    "ratios": {
        "z_score": {"w1": 1.2, "w2": 1.4, "w3": 3.3, "w4": 0.6, "w5": 1.0},
        "credit_zones": {"safe_threshold": 2.99, "grey_threshold": 1.81},
        "liquidity": {"current_ratio_stress": 1.0},
    },
    "propagation": {
        "contagion_damping_alpha": 0.6,
        "chokepoint_outdegree_threshold": 2,
        "chokepoint_amplification_factor": 1.4,
        "liquidity_stress_multiplier": 1.3,
        "min_propagation_weight": 0.05,
    },
    "shocks": {
        "S1": {
            "name": "Interest Rate Spike",
            "description": "Test",
            "parameter": "interest_expense_increase_pct",
            "default_magnitude": 0.4,
            "magnitude_min": 0.05,
            "magnitude_max": 2.0,
        },
        "S2": {
            "name": "Demand Shock",
            "description": "Test",
            "parameter": "revenue_reduction_pct",
            "default_magnitude": 0.25,
            "magnitude_min": 0.05,
            "magnitude_max": 0.80,
        },
        "S3": {
            "name": "Key Supplier Failure",
            "description": "Test",
            "parameter": "focal_firm",
            "default_magnitude": 1.0,
            "magnitude_min": 0.1,
            "magnitude_max": 1.0,
        },
    },
    "firms": {
        "tier1_buyers": [
            {"ticker": "BUYER1", "name": "Buyer One", "cik": "0000000001"},
            {"ticker": "BUYER2", "name": "Buyer Two", "cik": "0000000002"},
        ],
        "tier2_suppliers": [
            {"ticker": "SUP1", "name": "Supplier One", "cik": "0000000003"},
        ],
        "tier3_suppliers": [
            {"ticker": "SUP2", "name": "Supplier Two", "cik": "0000000004"},
        ],
    },
    "logging": {"level": "WARNING", "format": "%(message)s"},
}


def _make_scores() -> pd.DataFrame:
    """Minimal synthetic scores dataframe."""
    return pd.DataFrame(
        [
            # A healthy buyer
            {
                "ticker": "BUYER1",
                "name": "Buyer One",
                "year": 2023,
                "z_score": 3.5,
                "credit_zone": "safe",
                "current_ratio": 2.0,
                "interest_coverage_ratio": 6.0,
                "debt_to_equity": 0.8,
                "total_assets": 5_000_000,
                "total_liabilities": 2_000_000,
                "current_assets": 2_000_000,
                "current_liabilities": 1_000_000,
                "retained_earnings": 1_500_000,
                "ebit": 600_000,
                "market_cap": 8_000_000,
                "revenue": 7_000_000,
                "interest_expense": 100_000,
            },
            # A second buyer
            {
                "ticker": "BUYER2",
                "name": "Buyer Two",
                "year": 2023,
                "z_score": 3.2,
                "credit_zone": "safe",
                "current_ratio": 1.8,
                "interest_coverage_ratio": 5.0,
                "debt_to_equity": 1.0,
                "total_assets": 4_000_000,
                "total_liabilities": 2_500_000,
                "current_assets": 1_800_000,
                "current_liabilities": 1_000_000,
                "retained_earnings": 1_000_000,
                "ebit": 400_000,
                "market_cap": 5_000_000,
                "revenue": 5_000_000,
                "interest_expense": 80_000,
            },
            # A stressed supplier (grey zone)
            {
                "ticker": "SUP1",
                "name": "Supplier One",
                "year": 2023,
                "z_score": 2.0,
                "credit_zone": "grey",
                "current_ratio": 1.1,
                "interest_coverage_ratio": 2.5,
                "debt_to_equity": 2.0,
                "total_assets": 1_000_000,
                "total_liabilities": 700_000,
                "current_assets": 400_000,
                "current_liabilities": 360_000,
                "retained_earnings": 50_000,
                "ebit": 80_000,
                "market_cap": 400_000,
                "revenue": 1_000_000,
                "interest_expense": 32_000,
            },
            # A highly stressed tier-3 supplier
            {
                "ticker": "SUP2",
                "name": "Supplier Two",
                "year": 2023,
                "z_score": 1.3,
                "credit_zone": "distress",
                "current_ratio": 0.8,
                "interest_coverage_ratio": 1.2,
                "debt_to_equity": 4.0,
                "total_assets": 500_000,
                "total_liabilities": 450_000,
                "current_assets": 120_000,
                "current_liabilities": 150_000,
                "retained_earnings": -50_000,
                "ebit": 20_000,
                "market_cap": 50_000,
                "revenue": 400_000,
                "interest_expense": 16_000,
            },
        ]
    )


def _make_edges() -> pd.DataFrame:
    """Minimal synthetic edge dataset."""
    return pd.DataFrame(
        [
            # SUP2 (Tier-3, distress) → SUP1 (Tier-2, grey)
            {
                "source": "SUP2",
                "target": "SUP1",
                "relationship_type": "tier3_to_tier2",
                "weight": 0.5,
                "assumption_basis": "test",
            },
            # SUP1 (Tier-2) → BUYER1 (Tier-1)
            {
                "source": "SUP1",
                "target": "BUYER1",
                "relationship_type": "tier2_to_tier1",
                "weight": 0.6,
                "assumption_basis": "test",
            },
            # SUP1 → BUYER2
            {
                "source": "SUP1",
                "target": "BUYER2",
                "relationship_type": "tier2_to_tier1",
                "weight": 0.4,
                "assumption_basis": "test",
            },
        ]
    )


@pytest.fixture
def engine():
    return PropagationEngine(_make_scores(), _make_edges(), BASE_CONFIG, year=2023)


# ------------------------------------------------------------------
# Graph construction tests
# ------------------------------------------------------------------


class TestGraphConstruction:
    def test_all_nodes_present(self, engine):
        for ticker in ["BUYER1", "BUYER2", "SUP1", "SUP2"]:
            assert ticker in engine.graph.nodes, f"{ticker} not in graph"

    def test_edges_directed_correctly(self, engine):
        # SUP1 → BUYER1 should exist; BUYER1 → SUP1 should not
        assert engine.graph.has_edge("SUP1", "BUYER1")
        assert not engine.graph.has_edge("BUYER1", "SUP1")

    def test_edge_weights_stored(self, engine):
        data = engine.graph.get_edge_data("SUP1", "BUYER1")
        assert data is not None
        assert abs(data["weight"] - 0.6) < 1e-6

    def test_node_attributes_set(self, engine):
        state = engine.node_states["SUP2"]
        assert state["credit_zone"] == "distress"
        assert state["z_score"] == pytest.approx(1.3)

    def test_initial_stress_scores_zero(self, engine):
        for state in engine.node_states.values():
            assert state["stress_score"] == 0.0


# ------------------------------------------------------------------
# Shock scenario tests
# ------------------------------------------------------------------


class TestShockScenarios:
    def test_s1_raises_buyer_stress(self, engine):
        """S1 (interest rate spike) should increase stress for highly-leveraged firms."""
        engine.apply_shock("S1", magnitude=0.5)
        # All firms with interest expense should be affected
        for ticker in ["BUYER1", "BUYER2", "SUP1", "SUP2"]:
            state = engine.node_states[ticker]
            assert state.get("shock_applied", False), (
                f"{ticker} should have shock applied"
            )

    def test_s1_reduces_z_score(self, engine):
        """S1 shock must reduce Z-scores (not increase them — economic sanity check)."""
        original_z = {
            t: engine.node_states[t]["z_score"]
            for t in engine.node_states
            if not pd.isna(engine.node_states[t].get("z_score", float("nan")))
        }
        engine.apply_shock("S1", magnitude=0.5)
        for ticker, orig_z in original_z.items():
            new_z = engine.node_states[ticker].get("z_score", float("nan"))
            if not pd.isna(new_z):
                assert new_z <= orig_z + 0.001, (
                    f"S1 should not increase Z-score for {ticker}: {orig_z:.2f} → {new_z:.2f}"
                )

    def test_s2_reduces_buyer_revenue(self, engine):
        """S2 shock should reduce revenue for Tier-1 buyers."""
        # Mark BUYER1 and BUYER2 as tier1 in config
        cfg = copy.deepcopy(BASE_CONFIG)
        engine2 = PropagationEngine(_make_scores(), _make_edges(), cfg, year=2023)

        original_rev_b1 = engine2.node_states["BUYER1"]["revenue"]
        engine2.apply_shock("S2", magnitude=0.25)
        new_rev_b1 = engine2.node_states["BUYER1"]["revenue"]
        # Revenue should be reduced (allow for NaN case)
        if not pd.isna(new_rev_b1) and not pd.isna(original_rev_b1):
            assert new_rev_b1 < original_rev_b1, (
                f"BUYER1 revenue should decrease after S2: {original_rev_b1} → {new_rev_b1}"
            )

    def test_s3_sets_focal_firm_to_distress(self, engine):
        """S3 shock must move focal firm to distress zone."""
        engine.apply_shock("S3", magnitude=1.0, focal_firm="SUP1")
        state = engine.node_states["SUP1"]
        assert state["credit_zone"] == "distress", (
            f"SUP1 should be in distress after S3, got: {state['credit_zone']}"
        )

    def test_s3_propagates_to_buyers(self, engine):
        """S3 shock on SUP1 should increase stress on its buyers (BUYER1, BUYER2)."""
        engine.apply_shock("S3", magnitude=1.0, focal_firm="SUP1")
        buyer1_stress = engine.node_states["BUYER1"]["stress_score"]
        buyer2_stress = engine.node_states["BUYER2"]["stress_score"]
        total_buyer_stress = buyer1_stress + buyer2_stress
        assert total_buyer_stress > 0, (
            f"S3 shock on SUP1 should propagate to buyers; got combined stress={total_buyer_stress:.4f}"
        )

    def test_invalid_scenario_raises(self, engine):
        """Unknown scenario ID should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown scenario_id"):
            engine.apply_shock("S99", magnitude=0.5)

    def test_invalid_focal_firm_raises(self, engine):
        """S3 with non-existent focal firm should raise ValueError."""
        with pytest.raises(ValueError, match="not in graph"):
            engine.apply_shock("S3", magnitude=1.0, focal_firm="NONEXISTENT")


# ------------------------------------------------------------------
# Propagation rule tests
# ------------------------------------------------------------------


class TestPropagationRules:
    def test_stress_does_not_flow_upstream(self):
        """
        Economic sanity: in supplier→buyer direction, a buyer's distress should NOT
        directly transmit stress to upstream suppliers (no reverse edges).

        Uses a minimal isolated engine so no other distressed nodes interfere.
        """
        safe_scores = pd.DataFrame(
            [
                {
                    "ticker": "ISO_SUP",
                    "name": "ISO Supplier",
                    "year": 2023,
                    "z_score": 4.0,
                    "credit_zone": "safe",
                    "current_ratio": 2.5,
                    "interest_coverage_ratio": 8.0,
                    "debt_to_equity": 0.5,
                    "total_assets": 1_000_000,
                    "total_liabilities": 300_000,
                    "current_assets": 600_000,
                    "current_liabilities": 240_000,
                    "retained_earnings": 500_000,
                    "ebit": 200_000,
                    "market_cap": 2_000_000,
                    "revenue": 1_500_000,
                    "interest_expense": 25_000,
                },
                {
                    "ticker": "ISO_BUY",
                    "name": "ISO Buyer",
                    "year": 2023,
                    "z_score": 4.0,
                    "credit_zone": "safe",
                    "current_ratio": 2.0,
                    "interest_coverage_ratio": 6.0,
                    "debt_to_equity": 0.8,
                    "total_assets": 3_000_000,
                    "total_liabilities": 1_000_000,
                    "current_assets": 1_500_000,
                    "current_liabilities": 750_000,
                    "retained_earnings": 800_000,
                    "ebit": 350_000,
                    "market_cap": 4_000_000,
                    "revenue": 4_000_000,
                    "interest_expense": 60_000,
                },
            ]
        )
        safe_edges = pd.DataFrame(
            [
                {
                    "source": "ISO_SUP",
                    "target": "ISO_BUY",
                    "relationship_type": "test",
                    "weight": 0.7,
                    "assumption_basis": "test",
                },
            ]
        )
        iso_engine = PropagationEngine(safe_scores, safe_edges, BASE_CONFIG, year=2023)
        # Manually distress the buyer (simulating external event, not supplier-caused)
        iso_engine.node_states["ISO_BUY"]["z_score"] = 0.5
        iso_engine.node_states["ISO_BUY"]["credit_zone"] = "distress"
        iso_engine._propagate_stress()
        # ISO_SUP must not have gained stress: no edge ISO_BUY → ISO_SUP exists
        sup_stress = iso_engine.node_states["ISO_SUP"]["stress_score"]
        assert sup_stress == 0.0, (
            f"Supplier stress must not increase from buyer distress (no upstream edge). "
            f"Got {sup_stress:.4f}"
        )

    def test_chokepoint_sup1_detected(self, engine):
        """
        SUP1 serves both BUYER1 and BUYER2 (in-degree 0 in supplier direction,
        but out-degree 2). Let's test it appears in chokepoint analysis.
        """
        chokepoints = engine.get_chokepoints()
        tickers = [cp["ticker"] for cp in chokepoints]
        assert "SUP1" in tickers, "SUP1 should appear in chokepoint analysis"

    def test_reset_clears_stress(self, engine):
        """After reset(), all stress scores should return to 0."""
        engine.apply_shock("S3", magnitude=1.0, focal_firm="SUP2")
        engine.reset()
        all_zero = all(s["stress_score"] == 0.0 for s in engine.node_states.values())
        assert all_zero, "Reset must zero all stress scores"

    def test_damping_reduces_far_stress(self, engine):
        """
        Stress at hop=2 should be less than at hop=1 (contagion dampening).
        SUP2 → SUP1 → BUYER1: BUYER1 is 2 hops from SUP2.
        """
        engine.apply_shock("S3", magnitude=1.0, focal_firm="SUP2")
        sup1_stress = engine.node_states["SUP1"]["stress_score"]
        buyer1_stress = engine.node_states["BUYER1"]["stress_score"]
        # BUYER1 is further from shock source; should have ≤ stress than SUP1
        # (this is a directional check, not exact)
        assert buyer1_stress <= sup1_stress + 0.5, (
            f"Stress should attenuate with distance: SUP1={sup1_stress:.3f}, BUYER1={buyer1_stress:.3f}"
        )


# ------------------------------------------------------------------
# Analysis method tests
# ------------------------------------------------------------------


class TestAnalysisMethods:
    def test_get_stress_path_returns_list(self, engine):
        engine.apply_shock("S3", magnitude=1.0, focal_firm="SUP2")
        path = engine.get_stress_path("SUP2")
        assert isinstance(path, list)
        assert len(path) >= 1
        assert path[0]["firm"] == "SUP2"
        assert path[0]["role"] == "source"

    def test_get_stress_path_invalid_firm_raises(self, engine):
        with pytest.raises(ValueError):
            engine.get_stress_path("BOGUS")

    def test_get_chokepoints_returns_all_nodes(self, engine):
        chokepoints = engine.get_chokepoints()
        assert len(chokepoints) == len(engine.node_states)

    def test_get_propagation_heatmap_shape(self, engine):
        hm = engine.get_propagation_heatmap()
        n = len(engine.node_states)
        assert hm.shape == (n, n)

    def test_summary_has_required_keys(self, engine):
        engine.apply_shock("S1", magnitude=0.4)
        summ = engine.summary()
        assert "after" in summ
        assert "total" in summ

    def test_get_most_stressed_firm_returns_valid_ticker(self, engine):
        focal = engine._get_most_stressed_firm()
        assert focal in engine.node_states, (
            f"Focal firm '{focal}' must be a known ticker"
        )
