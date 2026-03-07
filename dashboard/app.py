"""
app.py — Financial Supply Chain Risk Platform — Streamlit Dashboard.

Run: streamlit run dashboard/app.py

Layout:
  Sidebar: sector, year, shock scenario, magnitude controls
  Tab 1:  Supply Chain Network (interactive Plotly graph)
  Tab 2:  Firm Detail (Z-score time series, ratios, exposure table)
  Tab 3:  Risk Summary (chokepoints, heatmap, before/after stats)
"""

import sys
import os
import logging
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yaml

# Ensure project root is in path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Supply Chain Risk Platform",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
CONFIG_PATH = ROOT / "config.yaml"


@st.cache_resource
def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


config = load_config()
dash_config = config["dashboard"]
ZONE_COLORS = dash_config["colors"]

# ------------------------------------------------------------------
# Data loaders (cached so they don't re-run on every interaction)
# ------------------------------------------------------------------


@st.cache_data
def load_macro() -> pd.DataFrame:
    """Load macroeconomic time series from disk (optional)."""
    path = ROOT / "data" / "macro" / "macro_series.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    return df


@st.cache_data
def load_scores() -> pd.DataFrame:
    """Load pre-computed risk scores from disk."""
    path = ROOT / "risk_framework" / "scores.csv"
    if not path.exists():
        st.error(
            "❌ scores.csv not found. Run `make scores` first to generate financial data."
        )
        st.stop()
    df = pd.read_csv(path)
    df["year"] = df["year"].astype(int)
    return df


@st.cache_data
def load_edges() -> pd.DataFrame:
    """Load supply chain edges from disk."""
    path = ROOT / "data" / "supply_chain" / "edges.csv"
    if not path.exists():
        st.error(
            "❌ edges.csv not found. Run `make supply_chain` first."
        )
        st.stop()
    return pd.read_csv(path)


@st.cache_resource
def get_engine(year: int):
    """Build PropagationEngine (cached per year)."""
    from propagation.propagation_engine import PropagationEngine
    scores_df = load_scores()
    edges_df = load_edges()
    return PropagationEngine(scores_df, edges_df, config, year)


# ------------------------------------------------------------------
# Sidebar Controls
# ------------------------------------------------------------------
st.sidebar.title("⚙️ Controls")
st.sidebar.markdown("---")

# Year selector
scores_df = load_scores()
available_years = sorted(scores_df["year"].unique().tolist(), reverse=True)
selected_year = st.sidebar.selectbox(
    "Analysis Year",
    available_years,
    index=0,
    help="Select the base year for financial data and risk scoring.",
)

# Shock scenario selector
scenario_options = {
    "None (baseline)": None,
    "S1: Interest Rate Spike": "S1",
    "S2: Demand Shock": "S2",
    "S3: Key Supplier Failure": "S3",
}
selected_scenario_label = st.sidebar.selectbox(
    "Shock Scenario",
    list(scenario_options.keys()),
    index=0,
)
selected_scenario = scenario_options[selected_scenario_label]

# Magnitude slider (context-sensitive)
magnitude = 0.0
focal_firm = None

if selected_scenario == "S1":
    magnitude = st.sidebar.slider(
        "Interest Expense Increase (%)",
        min_value=5, max_value=200, value=40, step=5,
        help="How much does interest expense increase? "
             "A 40% increase mimics the 2022 Fed rate hike cycle effect on floating-rate debt."
    ) / 100.0
    st.sidebar.info(
        "**Economic interpretation:** A 40% jump in interest expense mirrors the impact "
        "of the 2022 Fed Funds rate increase (0.25% → 5.25%) on firms with floating-rate debt. "
        "Firms with ICR < 2.0 are most vulnerable."
    )

elif selected_scenario == "S2":
    magnitude = st.sidebar.slider(
        "OEM Revenue Reduction (%)",
        min_value=5, max_value=80, value=25, step=5,
        help="By how much does Tier-1 buyer revenue fall? "
             "25% mirrors COVID-19 auto plant shutdowns (2020 Q2)."
    ) / 100.0
    st.sidebar.info(
        "**Economic interpretation:** A 25% demand shock at the OEM level "
        "mirrors COVID-19 production halts or a semiconductor shortage. "
        "Suppliers with high customer concentration are most exposed."
    )

elif selected_scenario == "S3":
    all_tickers = sorted(scores_df["ticker"].unique().tolist())
    focal_firm = st.sidebar.selectbox(
        "Focal Firm (Supplier in Distress)",
        all_tickers,
        help="Select the firm to set into financial distress.",
    )
    magnitude = st.sidebar.slider(
        "Distress Severity",
        min_value=0.1, max_value=1.0, value=1.0, step=0.1,
        help="1.0 = full bankruptcy; 0.5 = severe but survivable distress.",
    )
    st.sidebar.info(
        f"**Economic interpretation:** Setting {focal_firm} into distress models a "
        "supplier bankruptcy, major plant fire, or geopolitical supply disruption. "
        "Buyers with high dependency (thick edges) are most impacted."
    )

# Macro context (if available)
macro_df = load_macro()
if not macro_df.empty:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Macro Context")
    fred_map = config["data"]["fred_series"]
    latest_macro = macro_df.sort_values("date").groupby("series_id").last()
    for name, series_id in fred_map.items():
        if series_id in latest_macro.index:
            val = latest_macro.loc[series_id, "value"]
            label = name.replace("_", " ").title()
            st.sidebar.metric(label, f"{val:.2f}")

st.sidebar.markdown("---")
run_simulation = st.sidebar.button(
    "▶ Run Simulation",
    type="primary",
    use_container_width=True,
)

# ------------------------------------------------------------------
# Session State
# ------------------------------------------------------------------
if "engine" not in st.session_state or st.session_state.get("year") != selected_year:
    st.session_state["engine"] = get_engine(selected_year)
    st.session_state["year"] = selected_year
    st.session_state["node_states"] = None
    st.session_state["before_states"] = None
    st.session_state["shock_applied"] = False

engine = st.session_state["engine"]

if run_simulation and selected_scenario:
    import copy
    st.session_state["before_states"] = copy.deepcopy(engine.node_states)
    engine.reset()
    updated_states = engine.apply_shock(
        selected_scenario, magnitude, focal_firm=focal_firm
    )
    st.session_state["node_states"] = updated_states
    st.session_state["shock_applied"] = True
elif not st.session_state.get("shock_applied"):
    st.session_state["node_states"] = engine.node_states

node_states = st.session_state["node_states"] or engine.node_states

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.title("🏭 Financial Supply Chain Risk Platform")
st.markdown(
    f"**Sector:** {config['sector'].capitalize()}  |  "
    f"**Base Year:** {selected_year}  |  "
    f"**Scenario:** {selected_scenario_label}"
)

if st.session_state.get("shock_applied") and selected_scenario:
    shock_cfg = config["shocks"].get(selected_scenario, {})
    st.success(
        f"✅ Shock applied: **{shock_cfg.get('name', selected_scenario)}**  "
        f"(magnitude = {magnitude:.0%})"
    )
    with st.expander("📖 Economic Interpretation"):
        st.markdown(shock_cfg.get("description", ""))

st.markdown("---")

# ------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌐 Network View", "🏢 Firm Detail", "📊 Risk Summary",
    "📋 Case Studies", "💰 Trade Credit Exposure",
])

# ==================================================================
# TAB 1: Network View
# ==================================================================
with tab1:
    from dashboard.graph_utils import build_plotly_graph

    col1, col2 = st.columns([3, 1])
    with col1:
        # Highlight path for S3
        highlight_path = []
        if st.session_state.get("shock_applied") and selected_scenario == "S3" and focal_firm:
            try:
                path_data = engine.get_stress_path(focal_firm)
                highlight_path = [p["firm"] for p in path_data]
            except Exception:
                pass

        fig_network = build_plotly_graph(
            engine.graph, node_states, config, highlight_path
        )
        st.plotly_chart(fig_network, use_container_width=True, key="network_graph")

    with col2:
        st.markdown("### Legend")
        st.markdown("🟢 **Safe** — Z > 2.99")
        st.markdown("🟡 **Grey** — 1.81 < Z ≤ 2.99")
        st.markdown("🔴 **Distress** — Z ≤ 1.81")
        st.markdown("⚫ **Unknown** — Data gap")
        st.markdown("---")
        st.markdown("**Node size** = Total Assets")
        st.markdown("**Edge width** = Relationship weight")

        if st.session_state.get("shock_applied"):
            st.markdown("---")
            st.markdown("### After Shock")
            zone_counts = pd.Series({
                n: s.get("credit_zone", "unknown")
                for n, s in node_states.items()
            }).value_counts()
            for zone, count in zone_counts.items():
                color_map = {"safe": "🟢", "grey": "🟡", "distress": "🔴", "unknown": "⚫"}
                st.markdown(f"{color_map.get(zone, '⚫')} **{zone.capitalize()}**: {count} firms")

# ==================================================================
# TAB 2: Firm Detail
# ==================================================================
with tab2:
    from dashboard.graph_utils import build_z_score_timeseries, build_ratio_comparison

    col_left, col_right = st.columns([1, 2])

    with col_left:
        selected_firms = st.multiselect(
            "Select Firms for Comparison",
            options=sorted(scores_df["ticker"].unique().tolist()),
            default=scores_df["ticker"].unique().tolist()[:4],
            key="firm_selector",
        )

        if selected_firms:
            detail_ticker = st.selectbox(
                "Firm Detail View",
                selected_firms,
                key="detail_ticker",
            )

    with col_right:
        if selected_firms:
            fig_ts = build_z_score_timeseries(scores_df, selected_firms)
            st.plotly_chart(fig_ts, use_container_width=True, key="z_score_ts")

    st.markdown("---")

    if selected_firms:
        # Ratio bar chart for selected year
        fig_bar = build_ratio_comparison(scores_df, selected_year)
        st.plotly_chart(fig_bar, use_container_width=True, key="ratio_bar")

    # Upstream/Downstream exposure table for detail_ticker
    if selected_firms and "detail_ticker" in st.session_state:
        ticker = st.session_state["detail_ticker"]
        st.markdown(f"### 🔍 {ticker} — Supply Chain Exposure")

        edges_df = load_edges()

        # Upstream (suppliers of this firm)
        upstream = edges_df[edges_df["target"] == ticker]
        # Downstream (buyers of this firm)
        downstream = edges_df[edges_df["source"] == ticker]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**⬆ Upstream Suppliers**")
            if upstream.empty:
                st.info("No upstream suppliers in dataset.")
            else:
                up_display = upstream[["source", "weight", "relationship_type", "assumption_basis"]].copy()
                up_display.columns = ["Supplier", "Weight", "Type", "Basis"]
                # Add current zone for each supplier
                zones = []
                for sup in up_display["Supplier"]:
                    state = node_states.get(sup, {})
                    zones.append(state.get("credit_zone", "unknown"))
                up_display.insert(1, "Zone", zones)
                st.dataframe(up_display, use_container_width=True, hide_index=True)

        with c2:
            st.markdown("**⬇ Downstream Buyers**")
            if downstream.empty:
                st.info("No downstream buyers in dataset.")
            else:
                dn_display = downstream[["target", "weight", "relationship_type", "assumption_basis"]].copy()
                dn_display.columns = ["Buyer", "Weight", "Type", "Basis"]
                zones = []
                for buyer in dn_display["Buyer"]:
                    state = node_states.get(buyer, {})
                    zones.append(state.get("credit_zone", "unknown"))
                dn_display.insert(1, "Zone", zones)
                st.dataframe(dn_display, use_container_width=True, hide_index=True)

        # Shock impact table
        if st.session_state.get("shock_applied") and st.session_state.get("before_states"):
            st.markdown("#### Shock Impact")
            before_state = st.session_state["before_states"].get(ticker, {})
            after_state = node_states.get(ticker, {})

            metrics = ["z_score", "current_ratio", "interest_coverage_ratio", "stress_score"]
            impact_rows = []
            for m in metrics:
                b = before_state.get(m, float("nan"))
                a = after_state.get(m, float("nan"))
                delta = (a - b) if not (pd.isna(a) or pd.isna(b)) else float("nan")
                impact_rows.append({
                    "Metric": m.replace("_", " ").title(),
                    "Before": f"{b:.3f}" if not pd.isna(b) else "N/A",
                    "After": f"{a:.3f}" if not pd.isna(a) else "N/A",
                    "Change": f"{delta:+.3f}" if not pd.isna(delta) else "N/A",
                })
            st.dataframe(pd.DataFrame(impact_rows), use_container_width=True, hide_index=True)

# ==================================================================
# TAB 3: Risk Summary
# ==================================================================
with tab3:
    from dashboard.graph_utils import build_stress_heatmap

    # --- Before/After Zone Distribution ---
    st.markdown("### Zone Distribution: Before vs. After Shock")

    before_states = st.session_state.get("before_states") or engine.node_states
    summary = engine.summary(before_states)

    col_b, col_a = st.columns(2)

    with col_b:
        st.markdown("**Before Shock**")
        before_counts = summary.get("before", {})
        if not before_counts:
            before_counts = pd.Series({
                n: s.get("credit_zone", "unknown")
                for n, s in before_states.items()
            }).value_counts().to_dict()
        total = sum(before_counts.values()) or 1
        for zone in ["safe", "grey", "distress", "unknown"]:
            cnt = before_counts.get(zone, 0)
            color_map2 = {"safe": "🟢", "grey": "🟡", "distress": "🔴", "unknown": "⚫"}
            st.metric(
                label=f"{color_map2.get(zone, '⚫')} {zone.capitalize()}",
                value=f"{cnt} firms ({cnt/total*100:.0f}%)",
            )

    with col_a:
        st.markdown("**After Shock**")
        after_counts = summary.get("after", {})
        total_after = sum(after_counts.values()) or 1
        for zone in ["safe", "grey", "distress", "unknown"]:
            cnt_a = after_counts.get(zone, 0)
            cnt_b = before_counts.get(zone, 0)
            delta = cnt_a - cnt_b
            color_map2 = {"safe": "🟢", "grey": "🟡", "distress": "🔴", "unknown": "⚫"}
            st.metric(
                label=f"{color_map2.get(zone, '⚫')} {zone.capitalize()}",
                value=f"{cnt_a} firms ({cnt_a/total_after*100:.0f}%)",
                delta=f"{delta:+d} firms" if delta != 0 else "no change",
                delta_color="inverse" if zone in ("distress", "grey") else "normal",
            )

    st.markdown("---")

    # --- Chokepoint Table ---
    st.markdown("### 🎯 Chokepoint Analysis")
    st.markdown(
        "Firms ranked by composite risk score: structural centrality × financial stress. "
        "**Chokepoints** (🔴) have out-degree ≥ threshold AND are in grey/distress zone."
    )

    chokepoints = engine.get_chokepoints()
    if chokepoints:
        cp_df = pd.DataFrame(chokepoints)
        # Style the dataframe
        display_cols = [
            "ticker", "name", "in_degree", "credit_zone", "z_score",
            "stress_score", "betweenness_centrality", "risk_score", "is_chokepoint"
        ]
        available_display = [c for c in display_cols if c in cp_df.columns]
        cp_display = cp_df[available_display].copy()
        cp_display.columns = [c.replace("_", " ").title() for c in available_display]

        def style_zone(val):
            colors_map = {"safe": "#d5f5e3", "grey": "#fdebd0",
                          "distress": "#fadbd8", "unknown": "#f2f3f4"}
            return f"background-color: {colors_map.get(str(val).lower(), 'white')}"

        st.dataframe(cp_display, use_container_width=True, hide_index=True)

        # Highlight top chokepoints
        high_risk = [cp for cp in chokepoints if cp["is_chokepoint"]]
        if high_risk:
            st.warning(
                f"⚠️ **{len(high_risk)} critical chokepoint(s) identified:** "
                + ", ".join(f"**{cp['ticker']}**" for cp in high_risk)
            )
        else:
            st.success("✅ No critical chokepoints identified under current conditions.")

    st.markdown("---")

    # --- Stress Propagation Heatmap ---
    st.markdown("### 🔥 Stress Propagation Heatmap")
    st.markdown(
        "Each cell shows stress intensity transmitted from supplier (row) to buyer (column). "
        "Darker red = higher stress transmission through that edge."
    )

    heatmap_df = engine.get_propagation_heatmap()
    # Filter to non-zero rows/cols for readability
    nonzero_mask = (heatmap_df != 0).any(axis=1) | (heatmap_df != 0).any(axis=0)
    heatmap_filtered = heatmap_df.loc[
        (heatmap_df != 0).any(axis=1),
        (heatmap_df != 0).any(axis=0)
    ]

    if heatmap_filtered.empty:
        st.info("No stress transmission detected (run a shock scenario first).")
    else:
        fig_heatmap = build_stress_heatmap(heatmap_filtered)
        st.plotly_chart(fig_heatmap, use_container_width=True, key="heatmap")

    st.markdown("---")

    # --- Stress Path (for S3) ---
    if st.session_state.get("shock_applied") and selected_scenario == "S3" and focal_firm:
        st.markdown(f"### 📍 Stress Propagation Path from {focal_firm}")
        try:
            path_data = engine.get_stress_path(focal_firm)
            path_df = pd.DataFrame(path_data)
            st.dataframe(path_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Could not compute stress path: {e}")

    # --- Summary Statistics ---
    st.markdown("---")
    st.markdown("### 📈 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    year_data = scores_df[scores_df["year"] == selected_year]
    with col1:
        st.metric("Total Firms", scores_df["ticker"].nunique())
    with col2:
        avg_z = year_data["z_score"].mean()
        st.metric("Avg Z-Score", f"{avg_z:.2f}" if not pd.isna(avg_z) else "N/A")
    with col3:
        pct_dist = (year_data["credit_zone"] == "distress").mean() * 100
        st.metric("% in Distress", f"{pct_dist:.0f}%")
    with col4:
        avg_cr = year_data["current_ratio"].mean() if "current_ratio" in year_data.columns else float("nan")
        st.metric("Avg Current Ratio", f"{avg_cr:.2f}" if not pd.isna(avg_cr) else "N/A")

    # Macro time series chart
    if not macro_df.empty:
        st.markdown("---")
        st.markdown("### 📈 Macroeconomic Time Series")
        from dashboard.graph_utils import build_macro_chart
        fig_macro = build_macro_chart(macro_df, config)
        st.plotly_chart(fig_macro, use_container_width=True, key="macro_chart")

# ==================================================================
# TAB 4: Case Studies
# ==================================================================
with tab4:
    st.markdown("### 📋 Pre-Built Case Study Scenarios")
    st.markdown(
        "These case studies demonstrate how financial stress propagates through "
        "the automotive supply chain under documented real-world scenarios."
    )

    case_studies = {
        "BorgWarner Bankruptcy → Impact on Ford's Trade Credit": {
            "scenario": "S3",
            "focal_firm": "BWA",
            "magnitude": 1.0,
            "narrative": """
**Background:** BorgWarner supplies electrification and powertrain components to Ford (19% of
BWA revenue), Stellantis (14%), and GM (11%). A bankruptcy would immediately disrupt
EV drivetrain and turbocharger supply.

**Economic mechanism:**
1. Ford faces 40% edge-weighted exposure to BWA → emergency sourcing costs, production halts
2. Stellantis and GM face 30% and 25% exposure respectively
3. Working capital strain: Ford may need to pre-pay alternative suppliers
4. Trade credit write-off: any outstanding payables to BWA become claims in bankruptcy

**Key question for trade credit analysts:** What is the expected loss on receivables from
firms downstream of BWA?
            """,
        },
        "COVID-19 Demand Shock → Upstream Cascade to Tier-3": {
            "scenario": "S2",
            "focal_firm": None,
            "magnitude": 0.25,
            "narrative": """
**Background:** In Q2 2020, global auto production fell ~25% as OEM plants shut down.
This immediately reduced orders to Tier-2 and Tier-3 suppliers.

**Economic mechanism:**
1. OEM revenue drops 25% (Ford, GM, Toyota, Stellantis)
2. Operating leverage amplifies the EBIT impact (fixed costs remain constant)
3. Suppliers with high customer concentration (Lear: 24% GM, 20% Ford) face
   proportionally larger revenue hits
4. Smaller Tier-3 firms (Dana, Modine) have less financial buffer to absorb the shock

**Key question:** Which suppliers face liquidity crises, and which maintain safe Z-scores?
            """,
        },
        "2022 Rate Hike Cycle → Debt Service Stress": {
            "scenario": "S1",
            "focal_firm": None,
            "magnitude": 0.40,
            "narrative": """
**Background:** The Fed raised rates from 0.25% to 5.25% in 2022-2023, the fastest tightening
cycle in decades. Firms with floating-rate debt or maturing fixed-rate bonds faced sharply
higher interest expenses.

**Economic mechanism:**
1. Interest expense increases ~40% for firms with significant floating-rate exposure
2. ICR (Interest Coverage Ratio) falls — firms with ICR < 1.5 enter danger zone
3. EBIT is partially eroded as higher debt service costs reduce operating cash flow
4. Z-Score X3 (EBIT/Assets) deteriorates, potentially pushing grey-zone firms into distress

**Key question:** Which highly-leveraged suppliers (D/E > 3) cannot absorb the rate increase?
            """,
        },
    }

    selected_case = st.selectbox(
        "Select Case Study",
        list(case_studies.keys()),
        key="case_study_selector",
    )

    case = case_studies[selected_case]
    st.markdown(case["narrative"])

    if st.button("▶ Run This Case Study", key="run_case_study", type="primary"):
        import copy
        engine_case = get_engine(selected_year)
        before = copy.deepcopy(engine_case.node_states)
        engine_case.apply_shock(
            case["scenario"], case["magnitude"], focal_firm=case["focal_firm"]
        )
        case_summary = engine_case.summary(before)

        st.markdown("---")
        st.markdown("#### Results")

        c1, c2, c3 = st.columns(3)
        with c1:
            new_dist = case_summary.get("new_distressed", 0)
            st.metric("New Firms in Distress", f"+{new_dist}")
        with c2:
            pct = case_summary.get("pct_distress_after", 0)
            st.metric("% in Distress (After)", f"{pct:.0f}%")
        with c3:
            st.metric("Total Firms", case_summary.get("total", 0))

        # Show before/after table for all firms
        impact_rows = []
        for ticker in sorted(engine_case.node_states.keys()):
            b = before.get(ticker, {})
            a = engine_case.node_states.get(ticker, {})
            b_z = b.get("z_score", float("nan"))
            a_z = a.get("z_score", float("nan"))
            delta = (a_z - b_z) if not (pd.isna(a_z) or pd.isna(b_z)) else float("nan")
            impact_rows.append({
                "Firm": ticker,
                "Zone Before": b.get("credit_zone", "unknown"),
                "Z Before": f"{b_z:.2f}" if not pd.isna(b_z) else "N/A",
                "Zone After": a.get("credit_zone", "unknown"),
                "Z After": f"{a_z:.2f}" if not pd.isna(a_z) else "N/A",
                "ΔZ": f"{delta:+.2f}" if not pd.isna(delta) else "N/A",
                "Stress": f"{a.get('stress_score', 0):.3f}",
            })
        st.dataframe(pd.DataFrame(impact_rows), use_container_width=True, hide_index=True)

# ==================================================================
# TAB 5: Trade Credit Exposure
# ==================================================================
with tab5:
    from risk_framework.trade_credit import (
        compute_trade_credit_exposure, portfolio_summary
    )

    st.markdown("### 💰 Trade Credit Exposure Analysis")
    st.markdown(
        "Quantifies trade credit risk from the perspective of a financial institution "
        "providing working-capital financing or trade credit insurance to supply chain firms."
    )

    edges_df_tc = load_edges()
    exposure_df = compute_trade_credit_exposure(node_states, edges_df_tc)
    portfolio = portfolio_summary(exposure_df)

    # Portfolio-level metrics
    st.markdown("#### Portfolio Summary")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.metric(
            "Total AR at Risk",
            f"${portfolio['total_ar_at_risk']:,.0f}",
        )
    with p2:
        st.metric(
            "Expected Loss",
            f"${portfolio['total_expected_loss']:,.0f}",
        )
    with p3:
        st.metric("% AR at Risk", f"{portfolio['pct_ar_at_risk']:.1f}%")
    with p4:
        st.metric("Avg PD", f"{portfolio['avg_pd']:.1%}")

    st.markdown("---")

    # Firm-level exposure table
    st.markdown("#### Firm-Level Exposure")
    display_cols = [
        "ticker", "name", "credit_zone", "z_score", "pd_estimate",
        "accounts_receivable", "ar_at_risk", "expected_loss",
    ]
    available_cols = [c for c in display_cols if c in exposure_df.columns]
    exp_display = exposure_df[available_cols].copy()
    exp_display.columns = [c.replace("_", " ").title() for c in available_cols]

    # Format currency columns
    for col in ["Accounts Receivable", "Ar At Risk", "Expected Loss"]:
        if col in exp_display.columns:
            exp_display[col] = exp_display[col].apply(
                lambda v: f"${v:,.0f}" if not pd.isna(v) else "N/A"
            )
    if "Pd Estimate" in exp_display.columns:
        exp_display["Pd Estimate"] = exp_display["Pd Estimate"].apply(
            lambda v: f"{v:.1%}" if not pd.isna(v) else "N/A"
        )

    st.dataframe(exp_display, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(
        "**Methodology:** PD estimated from Altman Z-Score mapping. "
        "LGD = 60% (Moody's 2023 average for senior unsecured manufacturing claims). "
        "Expected Loss = AR × PD × LGD."
    )
