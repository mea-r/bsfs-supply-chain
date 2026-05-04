"""
app.py — Simplified Semiconductor Supply Chain Dashboard.

Run: streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import streamlit.components.v1 as components
from pyvis.network import Network

# Ensure project root is in path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.scenarios import run_idiosyncratic_shock, run_sector_shock, run_systemic_shock
from src.propagation_engine import compute_centrality, compute_hhi, compute_vulnerability, normalize_weight

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Semiconductor Supply Chain Risk",
    layout="wide",
)

# ------------------------------------------------------------------
# Data Loading
# ------------------------------------------------------------------

@st.cache_data
def load_nodes():
    path = ROOT / "data" / "Final Table (csv).csv"
    df = pd.read_csv(path)
    # Ensure Z'' is numeric
    df["Z''"] = pd.to_numeric(df["Z''"], errors='coerce')
    return df

@st.cache_data
def load_edges():
    path = ROOT / "data" / "Dependency relationships.csv"
    edges = pd.read_csv(path)
    
    # The IDs in Dependency relationships.csv don't match Final Table.
    # We must remap them using the Company names.
    nodes_df = load_nodes()
    name_map = {
        "Cadence": "Cadence Design Systems",
        "Siemens": "Siemens EDA (Mentor Graphics)",
        "Amkor": "Amkor Technology",
        "Axcelis": "Axcelis Technologies",
        "Globalfoundries": "GlobalFoundries",
        "Micron": "Micron Technology"
    }
    edges["company_a"] = edges["company_a"].replace(name_map)
    edges["company_b"] = edges["company_b"].replace(name_map)
    
    name_to_id = dict(zip(nodes_df["Company"], nodes_df["id"]))
    edges["a_id"] = edges["company_a"].map(name_to_id)
    edges["b_id"] = edges["company_b"].map(name_to_id)
    
    return edges

@st.cache_data
def compute_metrics(_G):
    centrality = compute_centrality(_G)
    hhi = compute_hhi(_G)
    vulnerability = compute_vulnerability(_G, hhi)
    return centrality, hhi, vulnerability

# ------------------------------------------------------------------
# Helper: Map Z'' to Zone Color
# ------------------------------------------------------------------
def get_zone_color(z):
    if pd.isna(z):
        return "#95a5a6"  # Unknown (Grey)
    if z > 2.6:
        return "#2ecc71"  # Safe (Green)
    if z > 1.1:
        return "#f39c12"  # Grey (Yellow)
    return "#e74c3c"      # Distress (Red)

def get_stress_color(stress):
    if pd.isna(stress): return "#95a5a6"
    if stress > 0.8: return "#e74c3c" # high stress
    if stress > 0.4: return "#f39c12" # medium
    return "#2ecc71" # low stress

# ------------------------------------------------------------------
# UI: Sidebar Settings & Filters
# ------------------------------------------------------------------
st.sidebar.title("Dashboard Settings")

# --- Visual Settings ---
with st.sidebar.expander("Visual Settings", expanded=True):
    show_labels = st.checkbox("Show Labels on Graph", value=True, help="Display company names permanently on the graph.")
    show_legend = st.checkbox("Show Risk Legend", value=True)
    physics_engine = st.selectbox(
        "Physics Engine", 
        ["Barnes-Hut (Constellation)", "ForceAtlas2 (Clustered)", "Repulsion (Spread out)"], 
        index=1  # Default to ForceAtlas2
    )
    layout_stiffness = st.slider("Node Separation / Spread", 0.5, 5.0, 1.0) # Default to 1.0

# --- Filters ---
st.sidebar.title("Filters")
nodes_df = load_nodes()
edges_df = load_edges()

# Value Chain Filter
with st.sidebar.expander("Value Chain Category", expanded=False):
    categories = sorted(nodes_df["Value Chain Category"].dropna().unique().tolist())
    cat_mode = st.radio("Category Filter Mode", ["All", "Custom"], index=0, key="cat_mode", horizontal=True)
    if cat_mode == "Custom":
        selected_categories = st.multiselect("Select Categories", categories, default=categories)
    else:
        selected_categories = categories

# Country Filter
with st.sidebar.expander("Country", expanded=False):
    countries = sorted(nodes_df["Country"].dropna().unique().tolist())
    country_mode = st.radio("Country Filter Mode", ["All", "Custom"], index=0, key="country_mode", horizontal=True)
    if country_mode == "Custom":
        selected_countries = st.multiselect("Select Countries", countries, default=countries)
    else:
        selected_countries = countries

# Apply filters
filtered_nodes = nodes_df[
    (nodes_df["Value Chain Category"].isin(selected_categories)) &
    (nodes_df["Country"].isin(selected_countries))
]

# --- Shock Scenarios ---
st.sidebar.title("Shock Scenarios")
with st.sidebar.expander("Scenario Configuration", expanded=False):
    scenario_type = st.selectbox("Scenario Type", ["None", "Idiosyncratic Shock", "Sector Shock", "Systemic Shock"])
    
    target_firm = None
    target_sector = None
    target_id = None
    if scenario_type == "Idiosyncratic Shock":
        target_firm = st.selectbox("Target Firm", sorted(nodes_df["Company"].dropna().unique().tolist()))
        firm_matches = nodes_df[nodes_df["Company"] == target_firm]
        if not firm_matches.empty:
            target_id = str(int(float(firm_matches["id"].iloc[0])))
    elif scenario_type == "Sector Shock":
        target_sector = st.selectbox("Target Sector", categories)
        
    shock_delta = st.slider("Shock Magnitude (Ds)", 0.0, 1.0, 0.35, 0.05)
    alpha_decay = st.slider("Propagation Alpha", 0.0, 1.0, 0.1, 0.05)
    run_shock = st.button("Run Simulation")

# ------------------------------------------------------------------
# Graph Construction
# ------------------------------------------------------------------

def build_network(filtered_nodes, edges_df):
    G = nx.DiGraph()

    filtered_nodes = filtered_nodes.dropna(subset=["id"])
    filtered_nodes["id"] = filtered_nodes["id"].apply(lambda x: str(int(float(x))))

    valid_ids = set(filtered_nodes["id"].tolist())

    for _, row in filtered_nodes.iterrows():
        G.add_node(
            row["id"],
            name=row["Company"],
            z_score=row["Z''"],
            category=row["Value Chain Category"],
            country=row["Country"],
            stress_baseline=row["Stress (logistic)"],
            stress=row["Stress (logistic)"]
        )

    for _, row in edges_df.iterrows():
        try:
            u = str(int(float(row["a_id"])))
            v = str(int(float(row["b_id"])))
        except (ValueError, TypeError):
            continue

        if u in valid_ids and v in valid_ids:
            strength = float(row["relationship_strength"])
            weight = normalize_weight(strength)
            is_sup = str(row.get("supplier", False)).strip().lower() == "true"
            is_cus = str(row.get("customer", False)).strip().lower() == "true"
            is_pat = str(row.get("partner", False)).strip().lower() == "true"
            
            if is_sup: G.add_edge(u, v, strength=strength, weight=weight)
            if is_cus: G.add_edge(v, u, strength=strength, weight=weight)
            if is_pat:
                G.add_edge(u, v, strength=strength/2.0, weight=weight/2.0)
                G.add_edge(v, u, strength=strength/2.0, weight=weight/2.0)

    # Remove isolated nodes
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)

    return G

G = build_network(filtered_nodes, edges_df)

# Remove firms from the dashboard data that don't have a connection
filtered_nodes = filtered_nodes[filtered_nodes["id"].apply(lambda x: str(int(float(x))) if pd.notna(x) else "").isin(G.nodes())]

# Compute metrics for the current graph
centrality, hhi, vulnerability = compute_metrics(G)

# Handle Simulation Execution
if "scenario_results" not in st.session_state:
    st.session_state.scenario_results = None

if run_shock:
    if scenario_type == "Idiosyncratic Shock" and target_id:
        st.session_state.scenario_results = run_idiosyncratic_shock(G, centrality, vulnerability, target_id, shock_delta, alpha_decay)
    elif scenario_type == "Sector Shock" and target_sector:
        st.session_state.scenario_results = run_sector_shock(G, centrality, vulnerability, target_sector, shock_delta, alpha_decay)
    elif scenario_type == "Systemic Shock":
        st.session_state.scenario_results = run_systemic_shock(G, centrality, vulnerability, shock_delta, alpha_decay)
    else:
        st.session_state.scenario_results = None

if scenario_type == "None":
    st.session_state.scenario_results = None

scenario_results = st.session_state.scenario_results

# ------------------------------------------------------------------
# Layout & Tabs
# ------------------------------------------------------------------
st.title("Semiconductor Supply Chain Risk Dashboard")

tab1, tab2 = st.tabs(["Network View", "Data Explorer"])

# ==================================================================
# TAB 1: Network View
# ==================================================================
with tab1:
    view_mode = "Before (Z'' Score)"
    if scenario_results:
        st.success(f"Simulation Active: {scenario_results['name']}")
        view_mode = st.radio("View Mode", ["Before (Z'' Score)", "After (Stress Score)"], horizontal=True)

    if show_legend:
        if view_mode == "Before (Z'' Score)":
            st.info("**Risk Zones (Z'' Score):** &nbsp;&nbsp; Safe (>2.6) &nbsp;&nbsp; | &nbsp;&nbsp; Grey (1.1-2.6) &nbsp;&nbsp; | &nbsp;&nbsp; Distress (≤1.1) &nbsp;&nbsp; | &nbsp;&nbsp; Unknown")
        else:
            st.info("**Stress Score:** &nbsp;&nbsp; Low (≤0.4) &nbsp;&nbsp; | &nbsp;&nbsp; Medium (0.4-0.8) &nbsp;&nbsp; | &nbsp;&nbsp; High (>0.8) &nbsp;&nbsp; | &nbsp;&nbsp; Unknown")

    if len(G.nodes) == 0:
        st.warning("No nodes match the selected filters.")
    else:
        # Build Pyvis Network (directed=False to remove arrows)
        net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="#333", directed=False)
        
        # Configure physics
        if "Barnes-Hut" in physics_engine:
            net.barnes_hut(
                gravity=-8000 * layout_stiffness, 
                central_gravity=0.3, 
                spring_length=150, 
                spring_strength=0.05, 
                damping=0.09, 
                overlap=0
            )
        elif "ForceAtlas2" in physics_engine:
            net.force_atlas_2based(
                gravity=-50 * layout_stiffness,
                central_gravity=0.01,
                spring_length=100,
                spring_strength=0.08,
                damping=0.4,
                overlap=0
            )
        else:
            net.repulsion(
                node_distance=150 * layout_stiffness,
                central_gravity=0.2,
                spring_length=200,
                spring_strength=0.05,
                damping=0.09
            )
            
        # Add nodes
        for node in G.nodes():
            attrs = G.nodes[node]
            
            if view_mode == "After (Stress Score)" and scenario_results:
                stress_val = scenario_results["stress_final"].get(node, attrs.get("stress", 0))
                color = get_stress_color(stress_val)
                delta = scenario_results['stress_change'].get(node, 0)
                title = f"<b>{attrs.get('name', str(node))}</b><br>Category: {attrs['category']}<br>Final Stress: {stress_val:.3f} (Δ {delta:+.3f})"
                
                # Highlight shocked nodes
                is_shocked = node in scenario_results.get("shocked_firms", [])
                border_width = 3 if is_shocked else 0
                border_color = "#000000" if is_shocked else color
            else:
                color = get_zone_color(attrs['z_score'])
                title = f"<b>{attrs.get('name', str(node))}</b><br>Category: {attrs['category']}<br>Z'' Score: {attrs['z_score']:.2f}"
                border_width = 0
                border_color = color

            # Pyvis workaround for border color
            color_dict = {
                "background": color,
                "border": border_color,
                "highlight": {
                    "background": color,
                    "border": border_color
                }
            }

            net.add_node(
                str(node),
                label=attrs.get("name", str(node)) if show_labels else " ",
                title=title,
                color=color_dict, 
                size=25,
                borderWidth=border_width,
                font={"size": 16, "color": "#333"}
            )
            
        # Add edges
        for u, v, data in G.edges(data=True):
            strength = data.get('strength', 1)

            u_clean = str(u).replace(".0", "")
            v_clean = str(v).replace(".0", "")

            if u_clean in net.get_nodes() and v_clean in net.get_nodes():
                width_map = {1: 1, 2: 2.5, 3: 5.0, 4: 8.5, 5: 13.0}
                width = width_map.get(int(strength), strength * 2)
                spring_len = max(30, 200 - (strength * 30))

                net.add_edge(
                    u, v,
                    value=width,
                    title=f"Strength: {strength}",
                    color={"color": "#b3b3b3", "highlight": "#555"},
                    length=spring_len
                )
            
        # Save and render interactively
        tmp_file = ROOT / "dashboard" / "pyvis_graph.html"
        net.save_graph(str(tmp_file))
        
        with open(tmp_file, 'r', encoding='utf-8') as f:
            source_code = f.read()
            
        components.html(source_code, height=710)

# ==================================================================
# TAB 2: Data Explorer
# ==================================================================
with tab2:
    if scenario_results:
        st.subheader("Simulation Results")
        st.write(f"**Scenario:** {scenario_results['name']} | **Rounds:** {scenario_results['rounds']}")
        
        stress_changes = scenario_results["stress_change"]
        affected_df = pd.DataFrame([
            {"Company": G.nodes[n].get("name", n), "Baseline Stress": G.nodes[n].get("stress", 0), "Final Stress": scenario_results["stress_final"][n], "Delta Stress": stress_changes[n]}
            for n in stress_changes
        ]).sort_values("Delta Stress", ascending=False).head(15)
        
        fig_sim = px.bar(
            affected_df, x="Delta Stress", y="Company", orientation='h',
            color="Final Stress", color_continuous_scale="Reds",
            title="Top Affected Firms (Δ Stress)"
        )
        fig_sim.update_layout(height=400, yaxis_autorange="reversed", yaxis_title=None)
        st.plotly_chart(fig_sim, use_container_width=True)
        st.markdown("---")

    # --- Metric Row ---
    m1, m2, m3, m4 = st.columns(4)
    avg_z = filtered_nodes["Z''"].mean()
    distress_count = len(filtered_nodes[filtered_nodes["Z''"] <= 1.1])
    m1.metric("Total Companies", len(filtered_nodes))
    m2.metric("Avg Z'' Score", f"{avg_z:.2f}")
    m3.metric("Firms in Distress", distress_count, delta=f"{distress_count/len(filtered_nodes):.0%}", delta_color="inverse")
    m4.metric("Categories", filtered_nodes["Value Chain Category"].nunique())

    st.markdown("---")

    # --- Row 1: Health & Industry ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Zone Distribution")
        def get_zone_label(z):
            if pd.isna(z): return "Unknown"
            if z > 2.6: return "Safe"
            if z > 1.1: return "Grey"
            return "Distress"
        
        plot_df = filtered_nodes.copy()
        plot_df["Zone"] = plot_df["Z''"].apply(get_zone_label)
        fig_pie = px.pie(
            plot_df, names="Zone", hole=0.4,
            color="Zone",
            color_discrete_map={"Safe": "#2ecc71", "Grey": "#f39c12", "Distress": "#e74c3c", "Unknown": "#95a5a6"}
        )
        fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Financial Health by Category")
        fig_box = px.box(
            filtered_nodes, x="Value Chain Category", y="Z''", 
            color="Value Chain Category",
            points="all",
            title="Z'' Score Distribution"
        )
        fig_box.update_layout(showlegend=False, height=350, xaxis_title=None)
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")

    # --- Row 2: Geography & Stress ---
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Geographic Risk (Avg Z'')")
        geo_df = filtered_nodes.groupby("Country")["Z''"].mean().reset_index().sort_values("Z''")
        fig_geo = px.bar(
            geo_df, x="Z''", y="Country", orientation='h',
            color="Z''", color_continuous_scale="RdYlGn",
            title="Lower is more risky"
        )
        fig_geo.update_layout(height=350, yaxis_title=None)
        st.plotly_chart(fig_geo, use_container_width=True)

    with col4:
        st.subheader("Top 10 High-Stress Companies (Baseline)")
        stress_df = filtered_nodes.sort_values("Stress (logistic)", ascending=False).head(10)
        fig_stress = px.bar(
            stress_df, x="Stress (logistic)", y="Company",
            orientation='h', color="Stress (logistic)", color_continuous_scale="Reds",
            text_auto='.3f'
        )
        fig_stress.update_layout(height=350, yaxis_autorange="reversed", yaxis_title=None)
        st.plotly_chart(fig_stress, use_container_width=True)

    st.markdown("---")

    # --- Row 3: Connectivity ---
    st.subheader("Supply Chain Hubs (Dependency Connectivity)")
    all_conns = pd.concat([edges_df["company_a"], edges_df["company_b"]])
    conn_counts = all_conns.value_counts().reset_index()
    conn_counts.columns = ["Company", "Connections"]
    conn_counts = conn_counts[conn_counts["Company"].isin(filtered_nodes["Company"])]
    
    fig_conn = px.bar(
        conn_counts.head(15), x="Connections", y="Company",
        orientation='h', color="Connections", color_continuous_scale="Purples",
        title="Most connected firms in the dataset"
    )
    fig_conn.update_layout(height=450, yaxis_autorange="reversed", yaxis_title=None)
    st.plotly_chart(fig_conn, use_container_width=True)

    st.markdown("---")
    
    # --- Data Tables ---
    with st.expander("Raw Company Data", expanded=False):
        st.dataframe(filtered_nodes.sort_values("Ranking"), use_container_width=True, hide_index=True)
    
    with st.expander("Raw Dependency Data", expanded=False):
        valid_companies = set(filtered_nodes["id"].tolist())
        display_edges = edges_df[
            (edges_df["company_a"].isin(valid_companies)) |
            (edges_df["company_b"].isin(valid_companies))
        ]
        st.dataframe(display_edges, use_container_width=True, hide_index=True)
