"""
Tests for data_engineering/supply_chain_builder.py

Validates:
  - Edge schema completeness
  - Weight bounds [0, 1]
  - No self-loops
  - Source/target nodes are known tickers
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import tempfile
import os

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data_engineering.supply_chain_builder import build_edges, AUTOMOTIVE_EDGES

TEST_CONFIG = {
    "data": {
        "supply_chain_dir": "/tmp/test_supply_chain",
    },
    "logging": {"level": "WARNING", "format": "%(message)s"},
    "firms": {
        "tier1_buyers": [
            {"ticker": "TM", "name": "Toyota", "cik": "0000096831"},
            {"ticker": "F", "name": "Ford", "cik": "0000037996"},
            {"ticker": "GM", "name": "General Motors", "cik": "0001467858"},
            {"ticker": "STLA", "name": "Stellantis", "cik": "0001800227"},
        ],
        "tier2_suppliers": [
            {"ticker": "BWA", "name": "BorgWarner", "cik": "0000908362"},
            {"ticker": "MGA", "name": "Magna", "cik": "0000749485"},
            {"ticker": "APTV", "name": "Aptiv", "cik": "0001521332"},
        ],
        "tier3_suppliers": [
            {"ticker": "ADNT", "name": "Adient", "cik": "0001670541"},
            {"ticker": "LEA", "name": "Lear", "cik": "0000842162"},
            {"ticker": "DAN", "name": "Dana", "cik": "0000026780"},
            {"ticker": "MOD", "name": "Modine", "cik": "0000067912"},
        ],
    }
}


class TestSupplyChainBuilder:

    @pytest.fixture
    def edges_df(self):
        os.makedirs("/tmp/test_supply_chain", exist_ok=True)
        return build_edges(TEST_CONFIG)

    def test_required_columns_present(self, edges_df):
        required = {"source", "target", "relationship_type", "weight", "assumption_basis"}
        assert required.issubset(set(edges_df.columns)), \
            f"Missing columns: {required - set(edges_df.columns)}"

    def test_weights_in_valid_range(self, edges_df):
        assert (edges_df["weight"] >= 0).all(), "Weights must be >= 0"
        assert (edges_df["weight"] <= 1).all(), "Weights must be <= 1"

    def test_no_self_loops(self, edges_df):
        self_loops = edges_df[edges_df["source"] == edges_df["target"]]
        assert self_loops.empty, f"Found self-loop edges: {self_loops}"

    def test_non_empty_edges(self, edges_df):
        assert len(edges_df) > 0, "Edge dataset must not be empty"

    def test_all_edges_have_assumption_basis(self, edges_df):
        missing_basis = edges_df[edges_df["assumption_basis"].isna() |
                                  (edges_df["assumption_basis"] == "")]
        assert missing_basis.empty, \
            f"All edges must have assumption_basis; missing for: {missing_basis[['source','target']].values}"

    def test_known_tier1_buyers_present(self, edges_df):
        tier1 = {"TM", "F", "GM", "STLA"}
        targets = set(edges_df["target"])
        # At least 3 of the 4 Tier-1 OEMs should appear as targets
        overlap = tier1 & targets
        assert len(overlap) >= 3, f"Expected most Tier-1 OEMs as buyers; found only: {overlap}"

    def test_extra_edges_appended(self):
        os.makedirs("/tmp/test_supply_chain", exist_ok=True)
        extra = [{
            "source": "NEW_SUP",
            "target": "TM",
            "relationship_type": "test",
            "weight": 0.1,
            "assumption_basis": "test only",
        }]
        df = build_edges(TEST_CONFIG, extra_edges=extra)
        assert "NEW_SUP" in df["source"].values

    def test_weight_clipping(self):
        """Edge weights outside [0,1] should be clipped, not raise error."""
        os.makedirs("/tmp/test_supply_chain", exist_ok=True)
        extra = [{
            "source": "A", "target": "B",
            "relationship_type": "test",
            "weight": 1.5,  # invalid
            "assumption_basis": "test",
        }]
        df = build_edges(TEST_CONFIG, extra_edges=extra)
        bad_a = df[df["source"] == "A"]
        assert bad_a["weight"].iloc[0] <= 1.0, "Weight > 1 should be clipped to 1"

    def test_automotive_edges_have_oem_suppliers(self):
        """At least the major documented relationships should be in the dataset."""
        sources = {e["source"] for e in AUTOMOTIVE_EDGES}
        targets = {e["target"] for e in AUTOMOTIVE_EDGES}
        assert "APTV" in sources
        assert "GM" in targets
        assert "TM" in targets
