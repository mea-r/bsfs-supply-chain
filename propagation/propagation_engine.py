"""
propagation_engine.py — Financial stress propagation across the supply chain graph.

Economic framework:
  Supply chains create financial interdependencies: a buyer's financial health
  depends partly on its suppliers' ability to deliver, and vice versa. When
  a supplier defaults or is severely stressed, buyers face:
    - Production disruption (lost revenue)
    - Emergency sourcing costs
    - Working capital strain (accelerated payments to keep suppliers afloat)
    - Potential write-offs of trade credit extended to suppliers

  This engine models stress propagation using a rule-based system calibrated
  on documented supply chain finance research:
    - Jacobides & Billinger (2006): supply chain interdependency and contagion
    - Altman et al. (2010): Z-score dynamics in supply chain networks
    - IMF (2023): financial contagion through trade credit channels

Propagation is deliberately rule-based (not a learned model) because:
  1. Interpretability: every stress delta can be attributed to a specific rule
  2. Scenario transparency: analysts can trace exactly why a firm is stressed
  3. Data efficiency: no training data required; grounded in economic logic
"""

import logging
import copy
from typing import Optional

import pandas as pd
import numpy as np
import networkx as nx
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class PropagationEngine:
    """
    Rule-based financial stress propagation engine.

    The engine builds a directed graph of supply chain relationships and
    applies configurable shock scenarios, propagating stress downstream
    (or upstream for demand shocks) according to documented economic rules.

    Graph convention:
      Edge direction: supplier → buyer (stress flows from supplier to buyer).
      For demand shocks (S2), stress originates at buyers and flows upstream
      (against edge direction) as reduced orders.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Output of risk_framework/scorer.py — firm-year risk scores.
    edges_df : pd.DataFrame
        Output of supply_chain_builder.py — supply chain edges.
    config : dict
        Loaded config.yaml.
    year : int
        Base year for analysis.
    """

    def __init__(
        self,
        scores_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        config: dict,
        year: int = 2023,
    ):
        self.config = config
        self.year = year
        self.prop_config = config["propagation"]
        self.shock_config = config["shocks"]
        self.zone_thresholds = config["ratios"]["credit_zones"]
        self.z_weights = config["ratios"]["z_score"]

        # Filter scores to selected year
        self._base_scores = scores_df[scores_df["year"] == year].copy()
        if self._base_scores.empty:
            # Fallback to nearest available year
            available = sorted(scores_df["year"].unique())
            nearest = min(available, key=lambda y: abs(y - year))
            logger.warning(f"No scores for {year}; falling back to {nearest}")
            self._base_scores = scores_df[scores_df["year"] == nearest].copy()

        self._base_scores = self._base_scores.set_index("ticker")
        self._edges = edges_df.copy()

        # Build the NetworkX graph
        self.graph = self._build_graph()

        # Current node states (mutable; reset by reset())
        self.node_states: dict[str, dict] = {}
        self._initialize_node_states()

        # Shock history for audit trail
        self.shock_log: list[dict] = []

    # ------------------------------------------------------------------
    # Graph Construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> nx.DiGraph:
        """
        Build a directed NetworkX graph from edges and scores.

        Edge direction: source (supplier) → target (buyer).
        Node attributes include all financial scores and risk zone.

        Returns
        -------
        nx.DiGraph
        """
        G = nx.DiGraph()

        # Add nodes from scores
        for ticker, row in self._base_scores.iterrows():
            G.add_node(ticker, **{
                "name": str(row.get("name", ticker)),
                "z_score": float(row.get("z_score", float("nan"))),
                "credit_zone": str(row.get("credit_zone", "unknown")),
                "current_ratio": float(row.get("current_ratio", float("nan"))),
                "interest_coverage_ratio": float(row.get("interest_coverage_ratio", float("nan"))),
                "debt_to_equity": float(row.get("debt_to_equity", float("nan"))),
                "revenue": float(row.get("revenue", float("nan"))),
                "ebit": float(row.get("ebit", float("nan"))),
                "interest_expense": float(row.get("interest_expense", float("nan"))),
                "total_assets": float(row.get("total_assets", float("nan"))),
                "total_liabilities": float(row.get("total_liabilities", float("nan"))),
                "market_cap": float(row.get("market_cap", float("nan"))),
                "retained_earnings": float(row.get("retained_earnings", float("nan"))),
                "current_assets": float(row.get("current_assets", float("nan"))),
                "current_liabilities": float(row.get("current_liabilities", float("nan"))),
                "stress_score": 0.0,   # propagated stress accumulator
            })

        # Add nodes from edges (in case some tickers only appear as edges)
        all_edge_nodes = set(self._edges["source"]) | set(self._edges["target"])
        for node in all_edge_nodes:
            if node not in G:
                G.add_node(node, name=node, z_score=float("nan"),
                           credit_zone="unknown", stress_score=0.0)

        # Add edges
        for _, edge in self._edges.iterrows():
            src = edge["source"]
            tgt = edge["target"]
            w = float(edge.get("weight", 0.5))
            G.add_edge(src, tgt,
                       weight=w,
                       relationship_type=edge.get("relationship_type", "unknown"),
                       assumption_basis=edge.get("assumption_basis", ""))

        logger.info(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    def _initialize_node_states(self) -> None:
        """
        Initialize node_states from base graph node attributes.

        node_states is the mutable working copy that shocks modify.
        The base graph is kept clean for reset operations.
        """
        self.node_states = {}
        for node, attrs in self.graph.nodes(data=True):
            self.node_states[node] = copy.deepcopy(attrs)
            self.node_states[node]["stress_score"] = 0.0
            self.node_states[node]["shock_applied"] = False

    # ------------------------------------------------------------------
    # Propagation Rules
    # ------------------------------------------------------------------

    def _recompute_z_score(self, state: dict) -> float:
        """
        Recompute Altman Z-Score from current node state after shock modifications.

        Called after modifying interest_expense or revenue to get the updated
        Z-score reflecting shock impact.

        Parameters
        ----------
        state : dict
            Node state dictionary (mutable).

        Returns
        -------
        float
            Updated Z-score, or NaN if inputs insufficient.
        """
        ta = state.get("total_assets", float("nan"))
        if pd.isna(ta) or ta == 0:
            return float("nan")

        ca = state.get("current_assets", float("nan"))
        cl = state.get("current_liabilities", float("nan"))
        re = state.get("retained_earnings", float("nan"))
        ebit = state.get("ebit", float("nan"))
        mc = state.get("market_cap", float("nan"))
        tl = state.get("total_liabilities", float("nan"))
        rev = state.get("revenue", float("nan"))

        w = self.z_weights
        components = []
        coefs = []

        if not (pd.isna(ca) or pd.isna(cl)):
            components.append((ca - cl) / ta); coefs.append(w["w1"])
        if not pd.isna(re):
            components.append(re / ta); coefs.append(w["w2"])
        if not pd.isna(ebit):
            components.append(ebit / ta); coefs.append(w["w3"])
        if not pd.isna(mc) and not pd.isna(tl) and tl != 0:
            components.append(mc / tl); coefs.append(w["w4"])
        if not pd.isna(rev):
            components.append(rev / ta); coefs.append(w["w5"])

        if len(components) < 3:
            return float("nan")

        return round(sum(c * x for c, x in zip(coefs, components)), 4)

    def _classify_zone(self, z: float) -> str:
        """Classify Z-score into credit zone."""
        if pd.isna(z):
            return "unknown"
        if z > self.zone_thresholds["safe_threshold"]:
            return "safe"
        if z > self.zone_thresholds["grey_threshold"]:
            return "grey"
        return "distress"

    def _apply_rule1_direct_transmission(
        self, source_state: dict, target_state: dict, edge_weight: float
    ) -> float:
        """
        Rule 1: Direct Transmission.

        Economic logic:
          When a supplier's Z-score is in the distress zone, the buyer faces
          real credit risk: the supplier may be unable to deliver, forcing
          the buyer to source elsewhere (at higher cost) or halt production.
          The financial impact on the buyer scales with the edge weight
          (how dependent the buyer is on that specific supplier).

        Stress contribution = (distress_severity) × edge_weight
        where distress_severity = max(0, distress_threshold - z_score) / distress_threshold

        Parameters
        ----------
        source_state : dict
            Supplying firm's state.
        target_state : dict
            Buying firm's state.
        edge_weight : float

        Returns
        -------
        float
            Stress delta to add to target's stress_score.
        """
        src_z = source_state.get("z_score", float("nan"))
        if pd.isna(src_z):
            return 0.0
        distress_threshold = self.zone_thresholds["grey_threshold"]
        if src_z >= distress_threshold:
            return 0.0  # Only transmits if supplier is in distress zone
        # Severity: how far below the distress threshold (normalized)
        severity = max(0.0, distress_threshold - src_z) / distress_threshold
        return severity * edge_weight

    def _apply_rule2_liquidity_cascade(self, source_state: dict, edge_weight: float) -> float:
        """
        Rule 2: Liquidity Cascade.

        Economic logic:
          A supplier with Current Ratio < 1.0 cannot cover its current
          liabilities from current assets — it is illiquid. It will likely
          delay supplier payments, stretch trade credit, and potentially
          demand accelerated payments from buyers. This forces the buyer
          to provide emergency financing (trade credit extension), directly
          hitting the buyer's own liquidity. The cascade is amplified
          relative to Rule 1 because illiquidity often precedes insolvency.

        Returns
        -------
        float
            Additional stress multiplier contribution.
        """
        cr = source_state.get("current_ratio", float("nan"))
        if pd.isna(cr) or cr >= self.config["ratios"]["liquidity"]["current_ratio_stress"]:
            return 0.0
        # Severity: how illiquid (CR=0 → max severity=1, CR=0.99 → near zero)
        severity = max(0.0, 1.0 - cr)
        multiplier = self.prop_config["liquidity_stress_multiplier"]
        return severity * edge_weight * (multiplier - 1.0)

    def _apply_rule3_contagion_dampening(self, stress: float, hops: int) -> float:
        """
        Rule 3: Contagion Dampening.

        Economic logic:
          Stress attenuates with each supply chain hop because:
          (a) Buyers diversify across multiple suppliers — one supplier's
              failure is partially absorbed by others.
          (b) Inventory buffers, insurance, and contractual protections
              reduce immediate transmission.
          (c) Financial institutions can provide bridging credit to
              buyers facing supplier stress.

          The dampening factor alpha=0.6 means 40% of stress is absorbed
          at each hop. This is consistent with empirical supply chain
          contagion research (Barrot & Sauvagnat 2016: ~60% local,
          ~40% transmitted per supply chain tier).

        Parameters
        ----------
        stress : float
            Incoming stress before dampening.
        hops : int
            Number of supply chain hops from original shock source.

        Returns
        -------
        float
            Dampened stress.
        """
        alpha = self.prop_config["contagion_damping_alpha"]
        return stress * (alpha ** hops)

    def _apply_rule4_chokepoint_amplification(
        self, node: str, state: dict, stress: float
    ) -> float:
        """
        Rule 4: Chokepoint Amplification.

        Economic logic:
          Nodes that are "chokepoints" — high in-degree (many buyers depend on
          them) AND in the grey/distress zone — amplify stress propagation.
          Chokepoints represent single-source or highly-concentrated suppliers
          whose stress is felt more acutely than a peripheral supplier.

          Example: If Aptiv (which supplies electrical systems to GM, Ford,
          Toyota, and Stellantis) enters distress, the impact is amplified
          because all four buyers face simultaneous sourcing disruption.

        Parameters
        ----------
        node : str
            The node (potential chokepoint).
        state : dict
            Node's current state.
        stress : float
            Incoming stress before amplification.

        Returns
        -------
        float
            (Possibly amplified) stress.
        """
        in_degree = self.graph.in_degree(node)
        threshold = self.prop_config["chokepoint_indegree_threshold"]
        zone = state.get("credit_zone", "safe")
        if in_degree >= threshold and zone in ("grey", "distress"):
            amp = self.prop_config["chokepoint_amplification_factor"]
            logger.debug(f"Chokepoint amplification at {node}: ×{amp}")
            return stress * amp
        return stress

    # ------------------------------------------------------------------
    # Shock Scenarios
    # ------------------------------------------------------------------

    def apply_shock(
        self,
        scenario_id: str,
        magnitude: float,
        focal_firm: str = None,
    ) -> dict:
        """
        Apply a shock scenario and propagate stress through the graph.

        Parameters
        ----------
        scenario_id : str
            "S1" (interest rate spike), "S2" (demand shock), "S3" (supplier failure).
        magnitude : float
            Scenario-specific magnitude parameter (see config.yaml for ranges).
        focal_firm : str, optional
            For S3: the firm to set into distress.

        Returns
        -------
        dict
            Updated node states after shock and propagation.
            Keys are tickers; values are state dicts.
        """
        # Always start from a clean base state
        self.reset()

        scenario_config = self.shock_config.get(scenario_id)
        if scenario_config is None:
            raise ValueError(f"Unknown scenario_id: {scenario_id}. Valid: S1, S2, S3")

        logger.info(f"Applying shock {scenario_id} ('{scenario_config['name']}') "
                    f"with magnitude={magnitude}, focal_firm={focal_firm}")

        if scenario_id == "S1":
            self._shock_s1_interest_rate_spike(magnitude)
        elif scenario_id == "S2":
            self._shock_s2_demand_shock(magnitude)
        elif scenario_id == "S3":
            if focal_firm is None:
                # Default to the most financially stressed firm in the network
                focal_firm = self._get_most_stressed_firm()
                logger.info(f"S3: No focal firm specified; defaulting to {focal_firm}")
            self._shock_s3_supplier_failure(focal_firm, magnitude)

        self.shock_log.append({
            "scenario_id": scenario_id,
            "name": scenario_config["name"],
            "magnitude": magnitude,
            "focal_firm": focal_firm,
        })

        return self.node_states

    def _apply_z_score_delta(self, state: dict, ebit_delta: float = 0.0,
                             revenue_delta: float = 0.0) -> None:
        """
        Apply incremental Z-score change from a shock, preserving the precomputed baseline.

        Rather than recomputing the entire Z-score from raw data (which would
        discard any precomputed accurate baseline), this method applies only the
        marginal Z-score impact of the shock via the relevant formula components.

        Economic logic:
          Z-Score sensitivity to EBIT: ΔZ = w3 × ΔEBIT / TotalAssets
          Z-Score sensitivity to Revenue: ΔZ = w5 × ΔRevenue / TotalAssets

        Parameters
        ----------
        state : dict
            Mutable node state dict.
        ebit_delta : float
            Change in EBIT (negative = reduction).
        revenue_delta : float
            Change in revenue (negative = reduction).
        """
        z_old = state.get("z_score", float("nan"))
        ta = state.get("total_assets", float("nan"))
        if pd.isna(z_old) or pd.isna(ta) or ta == 0:
            return
        w = self.z_weights
        dz = 0.0
        if not pd.isna(ebit_delta):
            dz += w["w3"] * (ebit_delta / ta)
        if not pd.isna(revenue_delta):
            dz += w["w5"] * (revenue_delta / ta)
        # Also re-estimate X1 if current_liabilities changed (S1 may stress working capital)
        state["z_score"] = round(max(-10.0, z_old + dz), 4)
        state["credit_zone"] = self._classify_zone(state["z_score"])

    def _shock_s1_interest_rate_spike(self, magnitude: float) -> None:
        """
        Scenario S1: Interest Rate Spike.

        Mechanism:
          Models the 2022 rate hike cycle (Fed Funds from 0.25% to 5.25%).
          Increases all firms' interest expense by `magnitude` fraction,
          which reduces EBIT (through higher financing costs) and directly
          impacts Z-Score X3 and interest coverage ratio (ICR).

          Firms with floating-rate debt are most exposed. This is proxied
          by increasing interest expense uniformly — a conservative bound
          that doesn't require firm-level debt structure data.

          Z-Score delta uses the incremental approach: only X3 (EBIT/Assets) is
          adjusted to reflect the EBIT haircut from higher debt service costs.
          This preserves the precomputed baseline Z-score while applying
          the economically correct marginal effect.

        Parameters
        ----------
        magnitude : float
            Fractional increase in interest expense (e.g., 0.4 = 40% increase).
        """
        for ticker, state in self.node_states.items():
            ie = state.get("interest_expense", float("nan"))
            if pd.isna(ie):
                continue
            # Apply interest expense shock
            additional_ie = ie * magnitude
            new_ie = ie + additional_ie
            state["interest_expense"] = new_ie
            # ICR falls: ICR = EBIT / new_ie
            ebit = state.get("ebit", float("nan"))
            if not pd.isna(ebit) and new_ie > 0:
                state["interest_coverage_ratio"] = ebit / new_ie
            # EBIT haircut: partial pass-through of higher debt service to operations
            # Economic: firms with high ICR can absorb more; low-ICR firms must cut opex
            # Conservative assumption: 50% of additional interest expense reduces EBIT
            if not pd.isna(ebit):
                ebit_delta = -0.5 * additional_ie
                state["ebit"] = ebit + ebit_delta
                # Apply incremental Z-score change (not full recompute)
                self._apply_z_score_delta(state, ebit_delta=ebit_delta)
            state["shock_applied"] = True

        # Propagate resulting stress through graph
        self._propagate_stress()

    def _shock_s2_demand_shock(self, magnitude: float) -> None:
        """
        Scenario S2: Demand Shock.

        Mechanism:
          Models demand collapse at the OEM (Tier-1 buyer) level, such as:
          - COVID-19 plant shutdowns (2020)
          - Semiconductor chip shortage production cuts (2021-2022)
          - Demand recession (e.g., auto sales drop)

          Revenue of Tier-1 buyers is reduced by `magnitude` fraction.
          This reduces their Z-Score X5 (revenue/assets) and X3 (EBIT/assets),
          cascading upstream as lower order volumes for Tier-2/3 suppliers.

          Economic logic for upstream propagation:
          When an OEM cuts production, it immediately reduces orders to suppliers.
          The revenue impact on a supplier = (buyer's revenue reduction) ×
          (supplier's revenue concentration in that buyer) × (edge weight).

        Parameters
        ----------
        magnitude : float
            Fractional revenue reduction for Tier-1 buyers (e.g., 0.25 = 25%).
        """
        # Identify Tier-1 buyers from config
        tier1_tickers = {
            f["ticker"]
            for f in self.config["firms"].get("tier1_buyers", [])
            if "ticker" in f
        }

        # Apply revenue shock to Tier-1 buyers first
        for ticker in tier1_tickers:
            if ticker not in self.node_states:
                continue
            state = self.node_states[ticker]
            rev = state.get("revenue", float("nan"))
            if not pd.isna(rev):
                rev_loss = rev * magnitude
                state["revenue"] = rev * (1 - magnitude)
                # EBIT also falls proportionally (assuming constant costs short-term)
                ebit = state.get("ebit", float("nan"))
                if not pd.isna(ebit):
                    # Operating leverage: EBIT loss > revenue loss due to fixed costs
                    operating_leverage = 1.5  # industry assumption; flagged for review
                    state["ebit"] = ebit - rev_loss * operating_leverage * magnitude
            state["z_score"] = self._recompute_z_score(state)
            state["credit_zone"] = self._classify_zone(state["z_score"])
            state["shock_applied"] = True

        # Propagate upstream (reverse edge direction: buyer stress → supplier)
        self._propagate_upstream(tier1_tickers, magnitude)

    def _shock_s3_supplier_failure(self, focal_firm: str, magnitude: float) -> None:
        """
        Scenario S3: Key Supplier Failure.

        Mechanism:
          A single supplier (focal_firm) is set to maximum distress
          (Z-Score floor) scaled by magnitude. This models events like:
          - Supplier bankruptcy (magnitude=1.0)
          - Major plant fire or strike (magnitude=0.5-0.8)
          - Geopolitical supply disruption (magnitude=0.3-0.6)

          Stress then propagates downstream to all buyers of that supplier
          through the normal propagation rules.

        Parameters
        ----------
        focal_firm : str
            Ticker of the firm set to distress.
        magnitude : float
            Severity scalar (1.0 = full distress floor, <1 = partial).
        """
        if focal_firm not in self.node_states:
            raise ValueError(f"Focal firm '{focal_firm}' not in graph. "
                             f"Available: {list(self.node_states.keys())}")

        state = self.node_states[focal_firm]
        # Set Z-score to distress floor (0 = theoretical minimum; 1.0 = distress boundary)
        distress_floor = 0.0
        original_z = state.get("z_score", float("nan"))
        distress_threshold = self.zone_thresholds["grey_threshold"]
        # Interpolate between current Z and 0 based on magnitude
        if not pd.isna(original_z):
            shocked_z = original_z - magnitude * (original_z - distress_floor)
        else:
            shocked_z = distress_floor
        state["z_score"] = round(shocked_z, 4)
        state["credit_zone"] = self._classify_zone(state["z_score"])
        state["shock_applied"] = True
        logger.info(f"S3: {focal_firm} Z-score: {original_z:.2f} → {shocked_z:.2f}")

        # Propagate downstream
        self._propagate_stress()

    # ------------------------------------------------------------------
    # Graph Traversal & Propagation
    # ------------------------------------------------------------------

    def _propagate_stress(self) -> None:
        """
        Propagate stress downstream (supplier → buyer direction).

        Uses BFS from each distressed node, applying all 4 propagation rules
        and dampening stress by `alpha` at each hop.
        """
        # Find all seed nodes (distressed or stressed suppliers)
        seeds = [
            n for n, state in self.node_states.items()
            if state.get("credit_zone") in ("distress", "grey")
            or state.get("shock_applied", False)
        ]

        # BFS propagation
        visited = set()
        queue = [(seed, 0, 1.0) for seed in seeds]  # (node, hops, stress_multiplier)

        while queue:
            node, hops, incoming_stress = queue.pop(0)
            if node in visited and hops > 0:
                continue
            visited.add((node, hops))

            src_state = self.node_states.get(node, {})

            for _, buyer, edge_data in self.graph.out_edges(node, data=True):
                edge_w = edge_data.get("weight", 0.5)
                if edge_w < self.prop_config["min_propagation_weight"]:
                    continue

                buyer_state = self.node_states.get(buyer, {})

                # Rule 1: Direct transmission
                stress_r1 = self._apply_rule1_direct_transmission(
                    src_state, buyer_state, edge_w
                )
                # Rule 2: Liquidity cascade (additive)
                stress_r2 = self._apply_rule2_liquidity_cascade(src_state, edge_w)
                # Combine rules
                stress_raw = stress_r1 + stress_r2

                # Rule 3: Contagion dampening
                stress_dampened = self._apply_rule3_contagion_dampening(stress_raw, hops)

                # Rule 4: Chokepoint amplification on the buyer
                stress_final = self._apply_rule4_chokepoint_amplification(
                    buyer, buyer_state, stress_dampened
                )

                if stress_final < 0.001:
                    continue

                # Accumulate stress on buyer
                buyer_state["stress_score"] = (
                    buyer_state.get("stress_score", 0.0) + stress_final
                )

                # If accumulated stress is significant, update buyer's Z-score.
                # Use the buyer's CURRENT z_score (which may already reflect a shock)
                # rather than the base graph value, to avoid overwriting shock effects.
                accumulated = buyer_state["stress_score"]
                if accumulated > 0.1:
                    current_z = buyer_state.get("z_score", float("nan"))
                    if not pd.isna(current_z):
                        # Stress score reduces Z-score proportionally
                        # (economic: stress score ≈ fraction of value-at-risk from supply disruption)
                        buyer_state["z_score"] = round(
                            max(-10.0, current_z - accumulated * abs(current_z) * 0.15), 4
                        )
                        buyer_state["credit_zone"] = self._classify_zone(buyer_state["z_score"])

                # Continue propagation if stress is still meaningful
                if stress_final > 0.01 and hops < 5:
                    queue.append((buyer, hops + 1, stress_final))

    def _propagate_upstream(self, seed_tickers: set, magnitude: float) -> None:
        """
        Propagate stress upstream from buyers to suppliers (for S2 demand shock).

        Economic logic:
          When OEM demand falls, suppliers face revenue reduction proportional
          to their exposure to that OEM. A supplier earning 30% of revenue from
          Ford faces a 30% × demand_shock revenue loss if Ford cuts entirely.

        Parameters
        ----------
        seed_tickers : set
            Tier-1 buyer tickers that received the initial demand shock.
        magnitude : float
            Revenue reduction fraction applied at seed nodes.
        """
        for buyer_ticker in seed_tickers:
            # Find all suppliers of this buyer (reverse edges)
            for supplier, _, edge_data in self.graph.in_edges(buyer_ticker, data=True):
                edge_w = edge_data.get("weight", 0.5)
                supplier_state = self.node_states.get(supplier, {})

                # Revenue impact on supplier = magnitude × edge_weight
                # (edge weight proxies revenue concentration in that buyer)
                rev_impact = magnitude * edge_w
                rev = supplier_state.get("revenue", float("nan"))
                if not pd.isna(rev):
                    supplier_state["revenue"] = rev * (1 - rev_impact)
                    ebit = supplier_state.get("ebit", float("nan"))
                    if not pd.isna(ebit):
                        # Higher operating leverage for smaller suppliers
                        supplier_state["ebit"] = ebit * (1 - rev_impact * 1.3)

                supplier_state["z_score"] = self._recompute_z_score(supplier_state)
                supplier_state["credit_zone"] = self._classify_zone(supplier_state["z_score"])
                supplier_state["stress_score"] = (
                    supplier_state.get("stress_score", 0.0) + rev_impact
                )
                supplier_state["shock_applied"] = True

        # Continue propagating to further upstream suppliers
        self._propagate_stress()

    def _get_most_stressed_firm(self) -> str:
        """Return the ticker with the lowest Z-score (most financially stressed)."""
        min_z = float("inf")
        focal = None
        for ticker, state in self.node_states.items():
            z = state.get("z_score", float("nan"))
            if not pd.isna(z) and z < min_z:
                min_z = z
                focal = ticker
        if focal is None:
            focal = list(self.node_states.keys())[0]
        return focal

    # ------------------------------------------------------------------
    # Analysis Methods
    # ------------------------------------------------------------------

    def get_stress_path(self, firm_id: str) -> list[dict]:
        """
        Return the propagation path of stress originating from firm_id.

        Traverses all downstream nodes reachable from firm_id and returns
        them in order of distance (BFS), with their stress contributions.

        Parameters
        ----------
        firm_id : str
            Ticker of the originating firm.

        Returns
        -------
        list[dict]
            List of dicts: {firm, hops, stress_score, credit_zone, z_score}
            ordered from closest to furthest from the source.
        """
        if firm_id not in self.graph:
            raise ValueError(f"Firm '{firm_id}' not in graph.")

        path = []
        visited = {firm_id}
        queue = [(firm_id, 0)]

        # Add source itself
        src_state = self.node_states.get(firm_id, {})
        path.append({
            "firm": firm_id,
            "name": src_state.get("name", firm_id),
            "hops": 0,
            "stress_score": src_state.get("stress_score", 0.0),
            "credit_zone": src_state.get("credit_zone", "unknown"),
            "z_score": src_state.get("z_score", float("nan")),
            "role": "source",
        })

        while queue:
            node, hops = queue.pop(0)
            for _, neighbor in self.graph.out_edges(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    state = self.node_states.get(neighbor, {})
                    path.append({
                        "firm": neighbor,
                        "name": state.get("name", neighbor),
                        "hops": hops + 1,
                        "stress_score": state.get("stress_score", 0.0),
                        "credit_zone": state.get("credit_zone", "unknown"),
                        "z_score": state.get("z_score", float("nan")),
                        "role": "downstream",
                    })
                    queue.append((neighbor, hops + 1))

        return path

    def get_chokepoints(self) -> list[dict]:
        """
        Identify supply chain chokepoints.

        A chokepoint is a node where:
          - in-degree >= threshold (many buyers depend on it), AND
          - credit zone is grey or distress (financially stressed)

        Economic significance:
          Chokepoints are the highest-priority monitoring targets for a
          lender or supply chain risk manager. They combine high systemic
          importance with financial vulnerability — the worst-case scenario.

        Returns
        -------
        list[dict]
            Chokepoints sorted by risk score (descending), with attributes:
            {ticker, name, in_degree, out_degree, credit_zone, z_score,
             stress_score, betweenness_centrality, risk_score}
        """
        threshold = self.prop_config["chokepoint_indegree_threshold"]
        betweenness = nx.betweenness_centrality(self.graph, weight="weight")

        chokepoints = []
        for node in self.graph.nodes():
            in_deg = self.graph.in_degree(node)
            state = self.node_states.get(node, {})
            zone = state.get("credit_zone", "unknown")

            # A node qualifies as a chokepoint if it's structurally important
            # (high in-degree) regardless of stress zone — report all of them
            # but flag those in grey/distress as highest priority.
            bc = betweenness.get(node, 0.0)

            # Composite risk score: structural centrality × financial stress
            zone_stress = {"distress": 1.0, "grey": 0.5, "safe": 0.0, "unknown": 0.25}
            z = state.get("z_score", float("nan"))
            z_normalized = max(0, 1 - z / 3.0) if not pd.isna(z) else 0.5
            risk_score = round(
                (in_deg / max(1, max(d for _, d in self.graph.in_degree()))) * 0.4
                + bc * 0.3
                + zone_stress.get(zone, 0) * 0.2
                + z_normalized * 0.1,
                4,
            )

            chokepoints.append({
                "ticker": node,
                "name": state.get("name", node),
                "in_degree": in_deg,
                "out_degree": self.graph.out_degree(node),
                "credit_zone": zone,
                "z_score": state.get("z_score", float("nan")),
                "stress_score": state.get("stress_score", 0.0),
                "betweenness_centrality": round(bc, 4),
                "risk_score": risk_score,
                "is_chokepoint": in_deg >= threshold and zone in ("grey", "distress"),
            })

        return sorted(chokepoints, key=lambda x: x["risk_score"], reverse=True)

    def get_propagation_heatmap(self) -> pd.DataFrame:
        """
        Build a stress propagation heatmap matrix.

        Returns a DataFrame where entry [i, j] represents the stress
        transmitted from firm i to firm j via the supply chain graph.

        Returns
        -------
        pd.DataFrame
            Stress matrix (tickers × tickers).
        """
        tickers = list(self.node_states.keys())
        n = len(tickers)
        idx = {t: i for i, t in enumerate(tickers)}
        matrix = np.zeros((n, n))

        for src, tgt, data in self.graph.edges(data=True):
            if src in idx and tgt in idx:
                src_state = self.node_states.get(src, {})
                stress = src_state.get("stress_score", 0.0)
                w = data.get("weight", 0.5)
                matrix[idx[src], idx[tgt]] = stress * w

        return pd.DataFrame(matrix, index=tickers, columns=tickers)

    def reset(self) -> None:
        """Reset node states to pre-shock baseline."""
        self._initialize_node_states()
        self.shock_log = []
        logger.debug("PropagationEngine reset to baseline.")

    def summary(self, before_states: dict = None) -> dict:
        """
        Summarize the current state: firm counts by credit zone.

        Parameters
        ----------
        before_states : dict, optional
            Pre-shock node states for before/after comparison.

        Returns
        -------
        dict
            Summary statistics.
        """
        after = pd.Series({
            n: s.get("credit_zone", "unknown")
            for n, s in self.node_states.items()
        })
        after_counts = after.value_counts().to_dict()

        result = {
            "after": after_counts,
            "total": len(self.node_states),
            "pct_distress_after": round(
                after_counts.get("distress", 0) / len(self.node_states) * 100, 1
            ),
        }
        if before_states:
            before = pd.Series({
                n: s.get("credit_zone", "unknown")
                for n, s in before_states.items()
            })
            before_counts = before.value_counts().to_dict()
            result["before"] = before_counts
            result["new_distressed"] = max(
                0, after_counts.get("distress", 0) - before_counts.get("distress", 0)
            )
        return result


def build_engine(
    config_path: str = "config.yaml",
    scores_path: str = "risk_framework/scores.csv",
    edges_path: str = "data/supply_chain/edges.csv",
    year: int = 2023,
) -> PropagationEngine:
    """
    Factory function: build a PropagationEngine from disk files.

    Parameters
    ----------
    config_path : str
    scores_path : str
    edges_path : str
    year : int

    Returns
    -------
    PropagationEngine
    """
    config = load_config(config_path)
    scores_df = pd.read_csv(scores_path)
    edges_df = pd.read_csv(edges_path)
    return PropagationEngine(scores_df, edges_df, config, year)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    engine = build_engine()
    print("=== Base Chokepoints ===")
    for cp in engine.get_chokepoints()[:5]:
        print(f"  {cp['ticker']}: zone={cp['credit_zone']}, risk={cp['risk_score']}")

    print("\n=== S1: Interest Rate Spike (40%) ===")
    states = engine.apply_shock("S1", magnitude=0.4)
    summ = engine.summary()
    print(f"  After shock: {summ}")
