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
    initial_sidebar_state="collapsed"
)
st.markdown("""
    <style>
        .block-container {
            padding-top: 2.5rem !important;
            padding-bottom: 0rem !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
            max-width: 100% !important;
        }
        iframe {
            border: 1px solid #e0e0e0 !important;
            border-radius: 8px;
            background-color: #fafafa;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Data Loading
# ------------------------------------------------------------------

@st.cache_data
def load_nodes():
    path = ROOT / "data" / "Final Table (csv).csv"
    df = pd.read_csv(path)
    df["Z''"] = pd.to_numeric(df["Z''"], errors="coerce")
    return df

@st.cache_data
def load_edges():
    path = ROOT / "data" / "Dependency relationships.csv"
    edges = pd.read_csv(path)
    
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
# Helper: Map Stress to Color
# ------------------------------------------------------------------
def get_stress_color(stress):
    if pd.isna(stress): return "#95a5a6"
    if stress > 0.9: return "#e74c3c"
    if stress > 0.5: return "#f39c12"
    return "#2ecc71"

def get_delta_color_fill(delta, max_delta):
    if delta <= 0.001: return "#cccccc"
    ratio = min(delta / (max_delta + 1e-9), 1.0)
    r = 255
    g = int(204 * (1 - ratio))
    b = int(204 * (1 - ratio))
    return f"#{r:02x}{g:02x}{b:02x}"

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

    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)

    return G

# ------------------------------------------------------------------
# UI Layout
# ------------------------------------------------------------------
nodes_df = load_nodes()
edges_df = load_edges()

categories = sorted(nodes_df["Value Chain Category"].dropna().unique().tolist())
countries = sorted(nodes_df["Country"].dropna().unique().tolist())

left_col, right_col = st.columns([1, 3], gap="large")

with left_col:
    st.markdown("### Shocks")
    scenario_type = st.selectbox("Scenario Type", ["Idiosyncratic", "Sector", "Systemic"])
    
    target_firm = None
    target_sector = None
    target_id = None
    if scenario_type == "Idiosyncratic":
        target_firm = st.selectbox("Target Firm", sorted(nodes_df["Company"].dropna().unique().tolist()))
        firm_matches = nodes_df[nodes_df["Company"] == target_firm]
        if not firm_matches.empty:
            target_id = str(int(float(firm_matches["id"].iloc[0])))
    elif scenario_type == "Sector":
        target_sector = st.selectbox("Target Sector", categories)
        
    c_mag, c_alph = st.columns(2)
    shock_delta = c_mag.number_input("Magnitude", min_value=0.0, max_value=1.0, value=0.45, step=0.05, format="%.2f")
    alpha_decay = c_alph.number_input("Alpha", min_value=0.0, max_value=1.0, value=0.15, step=0.05, format="%.2f")
    
    cb1, cb2 = st.columns(2)
    run_shock = cb1.button("Run Simulation", use_container_width=True, type="primary")
    reset_sim = cb2.button("Reset", use_container_width=True)

    if reset_sim:
        st.session_state.scenario_results = None
        st.session_state.view_mode = "Baseline"
        st.rerun()

    st.markdown("### Quick Examples")
    ex1, ex2 = st.columns(2)
    quick_run = False
    if ex1.button("TSMC", use_container_width=True):
        scenario_type = "Idiosyncratic"
        tsmc_matches = nodes_df[nodes_df["Company"] == "TSMC"]
        if not tsmc_matches.empty:
            target_id = str(int(float(tsmc_matches["id"].iloc[0])))
            quick_run = True
            shock_delta = 0.5
            alpha_decay = 0.1
    if ex2.button("Foundry", use_container_width=True):
        scenario_type = "Sector"
        target_sector = "Foundry"
        quick_run = True
        shock_delta = 0.4
        alpha_decay = 0.1

    top_impacted_container = st.container()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Controls")
    
    with st.expander("Visual Settings", expanded=False):
        show_labels = st.checkbox("Show Labels on Graph", value=True)
        show_legend = st.checkbox("Show Risk Legend", value=True)
        physics_engine = st.selectbox(
            "Physics Engine", 
            ["Barnes-Hut (Constellation)", "ForceAtlas2 (Clustered)", "Repulsion (Spread out)"], 
            index=1
        )
        layout_stiffness = st.slider("Node Separation / Spread", 0.5, 5.0, 1.0)

    with st.expander("Filters", expanded=False):
        cat_mode = st.radio("Category Filter", ["All", "Custom"], index=0, horizontal=True)
        if cat_mode == "Custom":
            selected_categories = st.multiselect("Select Categories", categories, default=categories)
        else:
            selected_categories = categories
            
        country_mode = st.radio("Country Filter", ["All", "Custom"], index=0, horizontal=True)
        if country_mode == "Custom":
            selected_countries = st.multiselect("Select Countries", countries, default=countries)
        else:
            selected_countries = countries

    filtered_nodes = nodes_df[
        (nodes_df["Value Chain Category"].isin(selected_categories)) &
        (nodes_df["Country"].isin(selected_countries))
    ]

# Compute network and metrics based on filters
G = build_network(filtered_nodes, edges_df)
filtered_nodes = filtered_nodes[filtered_nodes["id"].apply(lambda x: str(int(float(x))) if pd.notna(x) else "").isin(G.nodes())]
centrality, hhi, vulnerability = compute_metrics(G)

if "scenario_results" not in st.session_state:
    st.session_state.scenario_results = None
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "Baseline"

if run_shock or quick_run:
    if scenario_type == "Idiosyncratic" and target_id:
        st.session_state.scenario_results = run_idiosyncratic_shock(G, centrality, vulnerability, target_id, shock_delta, alpha_decay)
        st.session_state.view_mode = "Impact"
    elif scenario_type == "Sector" and target_sector:
        st.session_state.scenario_results = run_sector_shock(G, centrality, vulnerability, target_sector, shock_delta, alpha_decay)
        st.session_state.view_mode = "Impact"
    elif scenario_type == "Systemic":
        st.session_state.scenario_results = run_systemic_shock(G, centrality, vulnerability, shock_delta, alpha_decay)
        st.session_state.view_mode = "Impact"
    else:
        st.session_state.scenario_results = None

scenario_results = st.session_state.scenario_results

with top_impacted_container:
    if scenario_results:
        st.markdown("---")
        st.markdown("### Top Impacted")
        stress_changes = scenario_results["stress_change"]
        affected_data = []
        for n in stress_changes:
            if stress_changes[n] > 0.001:
                affected_data.append({
                    "Firm": G.nodes[n].get("name", n),
                    "Δ Stress": stress_changes[n]
                })
        affected_df = pd.DataFrame(affected_data)
        if not affected_df.empty:
            affected_df = affected_df.sort_values("Δ Stress", ascending=False).head(5)
            fig_sim = px.bar(
                affected_df, x="Δ Stress", y="Firm", orientation="h",
                color_discrete_sequence=["#ef553b"]
            )
            fig_sim.update_layout(
                height=220, 
                margin=dict(t=0, b=0, l=0, r=0), 
                yaxis_autorange="reversed",
                xaxis_title="", 
                yaxis_title="",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            fig_sim.update_xaxes(showgrid=False, zeroline=False, tickvals=[0, 0.2, 0.4])
            fig_sim.update_yaxes(showgrid=False, zeroline=False)
            st.plotly_chart(fig_sim, use_container_width=True, config={'displayModeBar': False})
            
            st.markdown("<br>", unsafe_allow_html=True)
            total_firms = len(G.nodes)
            avg_s = sum(scenario_results["stress_final"].values()) / total_firms
            distressed = sum(1 for v in scenario_results["stress_final"].values() if v > 0.9)
            st.write(f"**Distressed Firms (>0.9):** {distressed}")
            st.write(f"**Average Network Stress:** {avg_s:.3f}")
        else:
            st.info("No propagation effects recorded.")

# ------------------------------------------------------------------
# Main Content
# ------------------------------------------------------------------
with right_col:
    tab1, tab2 = st.tabs(["Network View", "Data Explorer"])

    # ==================================================================
    # TAB 1: Network View
    # ==================================================================
    with tab1:
        view_mode = st.session_state.get("view_mode")
        if not view_mode:
            view_mode = "Baseline"

        if len(G.nodes) == 0:
            st.warning("No nodes match the selected filters.")
        else:
            net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="#333", directed=False)
            
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
                
            max_delta = 0
            if scenario_results:
                max_delta = max(list(scenario_results.get("stress_change", {}).values()) + [0])

            for node in G.nodes():
                attrs = G.nodes[node]
                
                stress_baseline = attrs.get("stress_baseline", 0)
                is_shocked = False
                
                if scenario_results:
                    stress_final = scenario_results["stress_final"].get(node, stress_baseline)
                    delta = scenario_results["stress_change"].get(node, 0)
                    is_shocked = node in scenario_results.get("shocked_firms", [])
                else:
                    stress_final = stress_baseline
                    delta = 0
                
                if view_mode == "Impact":
                    color = get_delta_color_fill(delta, max_delta) if scenario_results else "#cccccc"
                    title = f"<b>{attrs.get('name', str(node))}</b><br>Category: {attrs['category']}<br>Δ Stress: {delta:+.3f}"
                    border_width = 3 if is_shocked else 1
                    border_color = "#000000" if is_shocked else "#999999"
                elif view_mode == "Final":
                    color = get_stress_color(stress_final)
                    title = f"<b>{attrs.get('name', str(node))}</b><br>Category: {attrs['category']}<br>Final Stress: {stress_final:.3f} (Δ {delta:+.3f})"
                    border_width = 3 if is_shocked else 0
                    border_color = "#000000" if is_shocked else color
                else: # Baseline
                    color = get_stress_color(stress_baseline)
                    title = f"<b>{attrs.get('name', str(node))}</b><br>Category: {attrs['category']}<br>Baseline Stress: {stress_baseline:.3f}"
                    border_width = 0
                    border_color = color

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
                
            for u, v, data in G.edges(data=True):
                strength = data.get("strength", 1)

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
                
            tmp_file = ROOT / "dashboard" / "pyvis_graph.html"
            net.save_graph(str(tmp_file))
            
            with open(tmp_file, "r", encoding="utf-8") as f:
                source_code = f.read()
                
            if scenario_results:
                overlay_html = f"""
                <div style="position: absolute; top: 15px; left: 15px; font-family: sans-serif; font-size: 14px; font-weight: 600; color: #444; background-color: rgba(255, 255, 255, 0.85); padding: 6px 12px; border-radius: 4px; z-index: 10;">
                    Simulation Active: {scenario_results['name']}
                </div>
                """
                source_code = source_code.replace("<body>", f"<body>{overlay_html}")
                
            components.html(source_code, height=700)
            
            st.markdown("<br>", unsafe_allow_html=True)
            has_sim = scenario_results is not None
            
            c_tabs, c_leg = st.columns([1, 2])
            with c_tabs:
                st.segmented_control(
                    "View Mode", 
                    ["Baseline", "Impact", "Final"], 
                    key="view_mode", 
                    label_visibility="collapsed",
                    disabled=not has_sim,
                    selection_mode="single"
                )
            with c_leg:
                if show_legend:
                    if view_mode == "Impact":
                        st.markdown("<div style='text-align: right; padding-top: 5px; color: #555; font-size: 14px;'><b>Impact (Δ Stress):</b> &nbsp; Gray (No Change) &nbsp; → &nbsp; Red (High Change)</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='text-align: right; padding-top: 5px; color: #555; font-size: 14px;'><b>Stress Score:</b> &nbsp; Low (≤0.5) &nbsp; | &nbsp; Medium (0.5-0.9) &nbsp; | &nbsp; High (>0.9)</div>", unsafe_allow_html=True)

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
                affected_df, x="Delta Stress", y="Company", orientation="h",
                color="Final Stress", color_continuous_scale="Reds",
                title="Top Affected Firms (Δ Stress)"
            )
            fig_sim.update_layout(height=400, yaxis_autorange="reversed", yaxis_title=None)
            st.plotly_chart(fig_sim, use_container_width=True)
            st.markdown("---")

        m1, m2, m3, m4 = st.columns(4)
        avg_stress = filtered_nodes["Stress (logistic)"].mean()
        distress_count = len(filtered_nodes[filtered_nodes["Stress (logistic)"] > 0.9])
        m1.metric("Total Companies", len(filtered_nodes))
        m2.metric("Avg Stress", f"{avg_stress:.3f}")
        m3.metric("Firms in Distress (>0.9)", distress_count, delta=f"{distress_count/len(filtered_nodes):.0%}", delta_color="inverse")
        m4.metric("Categories", filtered_nodes["Value Chain Category"].nunique())

        st.markdown("---")

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Stress Level Distribution")
            def get_stress_label(s):
                if pd.isna(s): return "Unknown"
                if s > 0.9: return "High"
                if s > 0.5: return "Medium"
                return "Low"
            
            plot_df = filtered_nodes.copy()
            plot_df["Stress Level"] = plot_df["Stress (logistic)"].apply(get_stress_label)
            fig_pie = px.pie(
                plot_df, names="Stress Level", hole=0.4,
                color="Stress Level",
                color_discrete_map={"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c", "Unknown": "#95a5a6"}
            )
            fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.subheader("Stress Distribution by Category")
            fig_box = px.box(
                filtered_nodes, x="Value Chain Category", y="Stress (logistic)", 
                color="Value Chain Category",
                points="all",
                title="Baseline Stress Distribution"
            )
            fig_box.update_layout(showlegend=False, height=350, xaxis_title=None)
            st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("---")

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Geographic Risk (Avg Stress)")
            geo_df = filtered_nodes.groupby("Country")["Stress (logistic)"].mean().reset_index().sort_values("Stress (logistic)", ascending=False)
            fig_geo = px.bar(
                geo_df, x="Stress (logistic)", y="Country", orientation="h",
                color="Stress (logistic)", color_continuous_scale="Reds",
                title="Higher is more risky"
            )
            fig_geo.update_layout(height=350, yaxis_title=None, yaxis_autorange="reversed")
            st.plotly_chart(fig_geo, use_container_width=True)

        with col4:
            st.subheader("Top 10 High-Stress Companies (Baseline)")
            stress_df = filtered_nodes.sort_values("Stress (logistic)", ascending=False).head(10)
            fig_stress = px.bar(
                stress_df, x="Stress (logistic)", y="Company",
                orientation="h", color="Stress (logistic)", color_continuous_scale="Reds",
                text_auto=".3f"
            )
            fig_stress.update_layout(height=350, yaxis_autorange="reversed", yaxis_title=None)
            st.plotly_chart(fig_stress, use_container_width=True)

        st.markdown("---")

        st.subheader("Supply Chain Hubs (Dependency Connectivity)")
        all_conns = pd.concat([edges_df["company_a"], edges_df["company_b"]])
        conn_counts = all_conns.value_counts().reset_index()
        conn_counts.columns = ["Company", "Connections"]
        conn_counts = conn_counts[conn_counts["Company"].isin(filtered_nodes["Company"])]
        
        fig_conn = px.bar(
            conn_counts.head(15), x="Connections", y="Company",
            orientation="h", color="Connections", color_continuous_scale="Purples",
            title="Most connected firms in the dataset"
        )
        fig_conn.update_layout(height=450, yaxis_autorange="reversed", yaxis_title=None)
        st.plotly_chart(fig_conn, use_container_width=True)

        st.markdown("---")
        
        with st.expander("Raw Company Data", expanded=False):
            st.dataframe(filtered_nodes.sort_values("Ranking"), use_container_width=True, hide_index=True)
        
        with st.expander("Raw Dependency Data", expanded=False):
            valid_companies = set(filtered_nodes["id"].tolist())
            display_edges = edges_df[
                (edges_df["company_a"].isin(valid_companies)) |
                (edges_df["company_b"].isin(valid_companies))
            ]
            st.dataframe(display_edges, use_container_width=True, hide_index=True)
