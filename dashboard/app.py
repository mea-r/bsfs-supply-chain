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
        /* Smaller font size and word wrap for all selectbox dropdown option lists */
        div[role="listbox"] [role="option"],
        div[role="listbox"] li,
        div[role="listbox"] div {
            font-size: 12px !important;
            white-space: normal !important;
            word-break: break-word !important;
        }
        /* Allow selected value text in selectboxes to wrap onto multiple lines as well */
        div[data-testid="stSelectbox"] div {
            white-space: normal !important;
            word-break: break-word !important;
        }
        /* Smaller font size and comfortable max-width margins for the Explanation tab (first tab) */
        div[data-baseweb="tab-panel"]:first-of-type,
        div[role="tabpanel"]:first-of-type,
        div[data-testid="stTabPanel"] > div[role="tabpanel"]:first-of-type,
        div[data-testid="stTabPanel"] [data-testid="stVerticalBlock"]:first-of-type {
            max-width: 800px !important;
            margin-left: auto !important;
            margin-right: auto !important;
            padding-left: 10px !important;
            padding-right: 10px !important;
        }
        div[data-baseweb="tab-panel"]:first-of-type p,
        div[data-baseweb="tab-panel"]:first-of-type li,
        div[data-baseweb="tab-panel"]:first-of-type span,
        div[data-baseweb="tab-panel"]:first-of-type blockquote,
        div[role="tabpanel"]:first-of-type p,
        div[role="tabpanel"]:first-of-type li,
        div[role="tabpanel"]:first-of-type span,
        div[role="tabpanel"]:first-of-type blockquote,
        div[data-testid="stTabPanel"] [data-testid="stVerticalBlock"]:first-of-type p,
        div[data-testid="stTabPanel"] [data-testid="stVerticalBlock"]:first-of-type li,
        div[data-testid="stTabPanel"] [data-testid="stVerticalBlock"]:first-of-type span,
        div[data-testid="stTabPanel"] [data-testid="stVerticalBlock"]:first-of-type blockquote,
        div[role="tabpanel"]:first-of-type [data-testid="stMarkdownContainer"] p,
        div[role="tabpanel"]:first-of-type [data-testid="stMarkdownContainer"] li,
        div[role="tabpanel"]:first-of-type [data-testid="stMarkdownContainer"] span,
        div[role="tabpanel"]:first-of-type [data-testid="stMarkdownContainer"] blockquote {
            font-size: 12.5px !important;
            line-height: 1.6 !important;
        }
        div[data-baseweb="tab-panel"]:first-of-type h2,
        div[role="tabpanel"]:first-of-type h2,
        div[data-testid="stTabPanel"] [data-testid="stVerticalBlock"]:first-of-type h2,
        div[role="tabpanel"]:first-of-type [data-testid="stMarkdownContainer"] h2 {
            font-size: 20px !important;
            margin-top: 0px !important;
            margin-bottom: 1rem !important;
        }
        div[data-baseweb="tab-panel"]:first-of-type h3,
        div[role="tabpanel"]:first-of-type h3,
        div[data-testid="stTabPanel"] [data-testid="stVerticalBlock"]:first-of-type h3,
        div[role="tabpanel"]:first-of-type [data-testid="stMarkdownContainer"] h3 {
            font-size: 14.5px !important;
            margin-top: 1.5rem !important;
            margin-bottom: 0.5rem !important;
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
    stress = max(0.0, min(1.0, stress))
    if stress < 0.85:
        # Green to Yellow (stays greener for much longer)
        ratio = stress / 0.85
        r = int(46 + ratio * (241 - 46))
        g = int(204 + ratio * (196 - 204))
        b = int(113 + ratio * (15 - 113))
    else:
        # Yellow to Red (steep ramp at the top end)
        ratio = (stress - 0.85) / 0.15
        r = int(241 + ratio * (231 - 241))
        g = int(196 + ratio * (76 - 196))
        b = int(15 + ratio * (60 - 15))
    return f"#{r:02x}{g:02x}{b:02x}"

def get_delta_color_fill(delta, max_delta):
    if delta <= 0.001: return "#cccccc"
    # Strong non-linear scaling to exaggerate small but meaningful changes
    ratio = min((delta / (max_delta + 1e-9)) ** 0.35, 1.0)
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
            value_chain_category=row["Value Chain Category"],
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
sector_counts = nodes_df["Value Chain Category"].value_counts().to_dict()
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
        target_sector = st.selectbox(
            "Target Sector", 
            categories,
            format_func=lambda x: f"{x} ({sector_counts.get(x, 0)})"
        )
        
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
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Quick Examples")
    st.markdown(
        "<div style='font-size: 11px; color: #555; margin-bottom: 10px; line-height: 1.45; font-family: -apple-system, BlinkMacSystemFont, sans-serif;'>"
        "<b>1. ASML Lithography (Idiosyncratic)</b>: Stress originates from ASML as a critical lithography bottleneck.<br>"
        "<b>2. Silicon Wafer (Sector)</b>: Stress is introduced to major wafer suppliers, reflecting a disruption in upstream wafer production."
        "</div>",
        unsafe_allow_html=True
    )
    ex1, ex2 = st.columns(2)
    quick_run = False
    if ex1.button("ASML Lithography", use_container_width=True):
        scenario_type = "Idiosyncratic"
        asml_matches = nodes_df[nodes_df["Company"] == "ASML"]
        if not asml_matches.empty:
            target_id = str(int(float(asml_matches["id"].iloc[0])))
            quick_run = True
            shock_delta = 0.50
            alpha_decay = 0.10
    if ex2.button("Silicon Wafer", use_container_width=True):
        scenario_type = "Sector"
        target_sector = "Wafer Manufacturing"
        quick_run = True
        shock_delta = 0.40
        alpha_decay = 0.10

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
if "graph_filter_selection" not in st.session_state:
    st.session_state["graph_filter_selection"] = "Full Network"

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



# ------------------------------------------------------------------
# Main Content
# ------------------------------------------------------------------
with right_col:
    tab_explanation, tab_network, tab_analytical = st.tabs(["Explanation", "Network View", "Analytical Results"])
    tab1 = tab_network
    tab2 = tab_analytical
    tab3 = tab_explanation

    # ==================================================================
    # TAB 1: Network View
    # ==================================================================
    with tab1:
        view_mode = st.session_state.get("view_mode")
        if not view_mode:
            view_mode = "Baseline"
            
        # Retrieve filter selection from the dropdown below the graph
        graph_filter = st.session_state.get("graph_filter_selection", "Full Network")
        
        if scenario_results:
            raw_name = scenario_results.get('name', '')
            if "Idiosyncratic Shock:" in raw_name:
                try:
                    node_id = raw_name.split(": ")[-1].strip()
                    firm_name = G.nodes[node_id].get("name", node_id) if node_id in G.nodes else node_id
                    overlay_name = f"Idiosyncratic Shock: {firm_name}"
                except Exception:
                    overlay_name = raw_name
            else:
                overlay_name = raw_name
            
            st.markdown(f"""
            <div style="display: inline-block; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 13px; font-weight: 600; color: #333; background-color: rgba(255, 255, 255, 0.95); padding: 7px 12px; border: 1px solid #e0e0e0; border-radius: 6px; margin-bottom: 12px;">
                Simulation Active: <span style="color: #e74c3c; margin-left: 4px;">{overlay_name}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="display: inline-block; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 13px; font-weight: 600; color: #555; background-color: rgba(255, 255, 255, 0.95); padding: 7px 12px; border: 1px solid #e0e0e0; border-radius: 6px; margin-bottom: 12px;">
                Simulation Status: <span style="color: #777; margin-left: 4px; font-weight: normal;">No active simulation (Baseline)</span>
            </div>
            """, unsafe_allow_html=True)
            
        sub_G = G.copy()
        if scenario_results and graph_filter != "Full Network":
            if graph_filter in ["1st Degree", "2nd Degree"]:
                radius = 1 if graph_filter == "1st Degree" else 2
                shocked = scenario_results.get("shocked_firms", [])
                if shocked:
                    undir_G = G.to_undirected()
                    nodes_to_keep = set()
                    for f in shocked:
                        if f in undir_G:
                            ego = nx.ego_graph(undir_G, f, radius=radius)
                            nodes_to_keep.update(ego.nodes())
                    sub_G = G.subgraph(list(nodes_to_keep)).copy()

        if len(sub_G.nodes) == 0:
            st.warning("No nodes match the selected filters.")
        else:
            net = Network(height="650px", width="100%", bgcolor="#ffffff", font_color="#333", directed=False)
            
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

            for node in sub_G.nodes():
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
                    title = f"{attrs.get('name', str(node))}\nCategory: {attrs['category']}\nΔ Stress: {delta:+.3f}"
                    border_width = 3 if is_shocked else 1
                    border_color = "#000000" if is_shocked else "#999999"
                elif view_mode == "Final":
                    color = get_stress_color(stress_final)
                    title = f"{attrs.get('name', str(node))}\nCategory: {attrs['category']}\nFinal Stress: {stress_final:.3f} (Δ {delta:+.3f})"
                    border_width = 3 if is_shocked else 0
                    border_color = "#000000" if is_shocked else color
                else: # Baseline
                    color = get_stress_color(stress_baseline)
                    title = f"{attrs.get('name', str(node))}\nCategory: {attrs['category']}\nBaseline Stress: {stress_baseline:.3f}"
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
                    font={
                        "size": 18, 
                        "color": "#111111", 
                        "strokeWidth": 3,
                        "strokeColor": "rgba(255, 255, 255, 0.85)",
                        "face": "sans-serif"
                    }
                )
                
            for u, v, data in sub_G.edges(data=True):
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
            
            source_code = net.generate_html()
            
            # Inject interaction hover config and randomSeed for deterministic layout
            source_code = source_code.replace(
                '"interaction": {',
                '"layout": {"randomSeed": 42},\n    "interaction": {\n        "hover": true,\n        "tooltipDelay": 50,'
            )

            # Inject window/load listener to guarantee proper viewport fit (stops zooming in issues on switching tabs)
            fit_script = """network = new vis.Network(container, data, options);
            
            // Re-fit the network after physics stabilized
            network.once("stabilizationIterationsDone", function() {
                setTimeout(function() {
                    network.fit();
                }, 100);
            });
            
            // Listen for window/iframe load & resize events to guarantee proper viewport fit
            window.addEventListener("load", function() {
                setTimeout(function() {
                    network.fit();
                }, 200);
            });
            window.addEventListener("resize", function() {
                network.fit();
            });"""
            source_code = source_code.replace("network = new vis.Network(container, data, options);", fit_script)
            
            css_injection = """
            <style>
            .vis-network { outline: none !important; }
            .vis-tooltip {
                position: absolute;
                padding: 12px 14px;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                font-size: 14px;
                color: #333 !important;
                background-color: rgba(255, 255, 255, 0.98) !important;
                border: 1px solid #ddd !important;
                border-radius: 8px !important;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
                pointer-events: none;
                z-index: 10000;
                line-height: 1.6;
                white-space: pre-wrap !important;
                font-weight: 500;
            }
            </style>
            """
            source_code = source_code.replace("</head>", f"{css_injection}\n</head>")
                
            components.html(source_code, height=650)
            
            st.markdown("<br>", unsafe_allow_html=True)
            has_sim = scenario_results is not None
            
            c_tabs, c_filter, c_leg = st.columns([1.2, 1.2, 2.0])
            with c_tabs:
                st.segmented_control(
                    "View Mode", 
                    ["Baseline", "Impact", "Final"], 
                    key="view_mode", 
                    label_visibility="collapsed",
                    disabled=not has_sim,
                    selection_mode="single"
                )
            with c_filter:
                st.markdown("""
                <style>
                div[data-baseweb="tab-panel"] div[data-testid="stSelectbox"] {
                    max-width: 200px !important;
                    margin-top: -16px !important;
                }
                div[data-baseweb="tab-panel"] div[data-testid="stSelectbox"] [data-baseweb="select"] {
                    background-color: #ffffff !important;
                    border: 1px solid #ccd4dc !important;
                    border-radius: 8px !important;
                    height: 32px !important;
                    min-height: 32px !important;
                    display: flex !important;
                    align-items: center !important;
                }
                div[data-baseweb="tab-panel"] div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
                    background-color: transparent !important;
                    border: none !important;
                    height: 100% !important;
                    min-height: 100% !important;
                    display: flex !important;
                    align-items: center !important;
                }
                div[data-baseweb="tab-panel"] div[data-testid="stSelectbox"] [data-baseweb="select"] div {
                    text-align: center !important;
                    justify-content: center !important;
                    align-items: center !important;
                    font-size: 13.5px !important;
                }
                </style>
                """, unsafe_allow_html=True)
                st.selectbox(
                    "Filter Graph",
                    ["Full Network", "1st Degree", "2nd Degree"],
                    key="graph_filter_selection",
                    label_visibility="collapsed",
                    disabled=not has_sim
                )
            with c_leg:
                if show_legend:
                    if view_mode == "Impact":
                        st.markdown("<div style='text-align: right; padding-top: 5px; color: #555; font-size: 14px;'><b>Impact (Δ Stress):</b> &nbsp; Gray (No Change) &nbsp; → &nbsp; Bright Red (High Change)</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='text-align: right; padding-top: 5px; color: #555; font-size: 14px;'><b>Stress Score:</b> &nbsp; Green (Low) &nbsp; → &nbsp; Yellow (Med) &nbsp; → &nbsp; Red (High)</div>", unsafe_allow_html=True)

    # ==================================================================
    # TAB 2: Analytical Results
    # ==================================================================
    with tab2:
        if not scenario_results:
            st.info("Run a simulation to view analytical results.")
        else:
            st.subheader("Scenario Summary")
            st.markdown("<div style='font-size: 13px; color: #555; margin-bottom: 10px;'>A macro-level overview of the simulated financial stress shock.</div>", unsafe_allow_html=True)
            
            c1, c2, c3, c4 = st.columns([2.5, 1, 1.2, 1.5])
            stress_changes = scenario_results["stress_change"]
            final_stresses = scenario_results["stress_final"]
            
            avg_base = sum(G.nodes[n].get("stress_baseline", 0) for n in G.nodes) / len(G.nodes)
            avg_final = sum(final_stresses.values()) / len(G.nodes)
            max_final = max(final_stresses.values())
            distressed = sum(1 for v in final_stresses.values() if v > 0.9)
            distress_share = distressed / len(G.nodes)
            
            c1.metric("Scenario", scenario_results["name"])
            c2.metric("Propagation Rounds", scenario_results["rounds"])
            c3.metric("Avg Final Stress", f"{avg_final:.3f}", delta=f"{avg_final - avg_base:+.3f}", delta_color="inverse")
            c4.metric("Distressed Firms (>0.9)", f"{distressed} ({distress_share:.1%})")
            
            st.markdown("---")
            
            col_l, col_r = st.columns(2)
            
            with col_l:
                st.subheader("Largest Stress Increases After Shock")
                st.markdown("<div style='font-size: 13px; color: #555; margin-bottom: 10px;'>Firms exhibiting the greatest delta (Δ) in stress as a direct result of network propagation.</div>", unsafe_allow_html=True)
                
                affected_df = pd.DataFrame([
                    {"Company": G.nodes[n].get("name", n), "Final Stress": final_stresses[n], "Delta Stress": stress_changes[n]}
                    for n in stress_changes
                ]).sort_values("Delta Stress", ascending=False).head(10)
                
                fig_sim = px.bar(
                    affected_df, x="Delta Stress", y="Company", orientation="h",
                    color="Delta Stress", color_continuous_scale="Reds",
                )
                fig_sim.update_layout(height=400, yaxis_autorange="reversed", yaxis_title=None, margin=dict(l=0, r=0, t=20, b=0))
                st.plotly_chart(fig_sim, use_container_width=True)

            with col_r:
                st.subheader("Systemic Chokepoints")
                st.markdown("<div style='font-size: 13px; color: #555; margin-bottom: 10px;'>Firms structurally positioned as critical transmission hubs, ranked by dependency connectivity.</div>", unsafe_allow_html=True)
                
                choke_data = []
                for n in G.nodes:
                    out_w = sum(d.get('weight', 0) for u, v, d in G.out_edges(n, data=True))
                    in_w = sum(d.get('weight', 0) for u, v, d in G.in_edges(n, data=True))
                    choke_data.append({"Company": G.nodes[n].get("name", n), "Systemic Importance": out_w + in_w})
                
                choke_df = pd.DataFrame(choke_data).sort_values("Systemic Importance", ascending=False).head(10)
                
                fig_choke = px.bar(
                    choke_df, x="Systemic Importance", y="Company", orientation="h",
                    color="Systemic Importance", color_continuous_scale="Purples",
                )
                fig_choke.update_layout(height=400, yaxis_autorange="reversed", yaxis_title=None, margin=dict(l=0, r=0, t=20, b=0))
                st.plotly_chart(fig_choke, use_container_width=True)

            st.markdown("---")
            
            st.subheader("Exposure by Value Chain Segment")
            st.markdown("<div style='font-size: 13px; color: #555; margin-bottom: 10px;'>Average final stress distributed across semiconductor value-chain categories.</div>", unsafe_allow_html=True)
            
            cat_data = []
            for n in G.nodes:
                cat = G.nodes[n].get("category", "Unknown")
                cat_data.append({"Category": cat, "Final Stress": final_stresses.get(n, G.nodes[n].get("stress_baseline", 0))})
            
            cat_df = pd.DataFrame(cat_data).groupby("Category")["Final Stress"].mean().reset_index().sort_values("Final Stress", ascending=False)
            
            fig_cat = px.bar(
                cat_df, x="Final Stress", y="Category", orientation="h",
                color="Final Stress", color_continuous_scale="Reds",
            )
            fig_cat.update_layout(height=400, yaxis_autorange="reversed", yaxis_title=None, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_cat, use_container_width=True)
            
            st.markdown("---")

    # ==================================================================
    # TAB 3: Explanation
    # ==================================================================
    with tab3:
        # Use columns to create comfortable side margins in pure Streamlit
        col_l, col_c, col_r = st.columns([1, 4, 1])
        with col_c:
            st.markdown("<h2 style='font-weight: 700; margin-top: 0; font-size: 20px; font-family: -apple-system, BlinkMacSystemFont, sans-serif;'>Methodology and Dashboard Guide</h2>", unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size: 12.5px; line-height: 1.6; font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin-bottom: 12px; color: #333;">
            This interactive platform simulates how financial distress propagates through supply dependencies in the global semiconductor network. It serves as a scenario-analysis tool for stress-testing and systemic risk identification rather than a precise forecasting system.
            </div>
            """, unsafe_allow_html=True)
            
            st.download_button(
                label="Download Full Project Report",
                data="Full Project Report Content Placeholder",
                file_name="semiconductor_risk_report.pdf",
                mime="application/pdf",
                disabled=True,
                help="The comprehensive project report is currently being finalized and will be downloadable soon."
            )
            
            st.markdown("---")
            
            st.markdown("<h3 style='font-size: 14.5px; font-weight: 600; margin-top: 1.5rem; margin-bottom: 0.5rem; font-family: -apple-system, BlinkMacSystemFont, sans-serif; color: #111;'>1. Introduction</h3>", unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size: 12.5px; line-height: 1.6; font-family: -apple-system, BlinkMacSystemFont, sans-serif; color: #333;">
            Modern industrial supply networks are highly complex, interdependent systems. In the semiconductor industry, this complexity is magnified by extreme geographic concentration, capital-intensive manufacturing processes, and highly specialized, non-substitutable inputs. A financial disruption at a single firm can quickly ripple outward, causing supply chain bottlenecks, factory shutdowns, and systematic financial contagion across the entire sector.
            
            <p style="margin-top: 12px; margin-bottom: 4px; font-weight: 600; color: #222;">What are we trying to explore?</p>
            <ul style="margin-top: 0px; padding-left: 20px;">
                <li><b>Vulnerability Mapping</b>: How localized financial distress cascades through the global semiconductor value chain.</li>
                <li><b>Chokepoint Identification</b>: Which firms serve as critical systemic transmission hubs that amplify propagating stress.</li>
                <li><b>Scenario Sensitivity</b>: How different types of shocks (idiosyncratic, sector-specific, and systemic) impact overall network stability.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<h3 style='font-size: 14.5px; font-weight: 600; margin-top: 1.5rem; margin-bottom: 0.5rem; font-family: -apple-system, BlinkMacSystemFont, sans-serif; color: #111;'>2. Data Collection & Assumptions</h3>", unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size: 12.5px; line-height: 1.6; font-family: -apple-system, BlinkMacSystemFont, sans-serif; color: #333;">
            <p style="margin-top: 0px; margin-bottom: 4px; font-weight: 600; color: #222;">Firm Selection & Financial Health (Z'' Score):</p>
            <ul style="margin-top: 0px; padding-left: 20px; margin-bottom: 12px;">
                <li>A representative sample of the world's leading semiconductor firms was selected across various value-chain segments (e.g., EDA tools, equipment, wafer manufacturing, fabless, foundries, OSAT).</li>
                <li>Financial health is quantified using the Altman Z''-Score formula, customized for emerging markets and non-manufacturing firms.</li>
                <li>Z''-scores are mapped to a normalized Baseline Stress level between 0.0 (perfect health) and 1.0 (distress) using a logistic mapping function. Low Z''-scores (in the Distress Zone) correspond to high baseline stress.</li>
            </ul>
            
            <p style="margin-top: 8px; margin-bottom: 4px; font-weight: 600; color: #222;">Network Edges & Relationships:</p>
            <ul style="margin-top: 0px; padding-left: 20px; margin-bottom: 12px;">
                <li>Dependencies (directed edges) represent documented supplier-customer relationships.</li>
                <li><b>Directed Flow</b>: An edge points from a supplier to a customer. Financial stress propagates downstream (if a supplier fails, the customer suffers) and upstream (if a major customer faces stress, their purchasing capacity drops, impacting the supplier).</li>
                <li><b>Relationship Strength</b>: Edge weights are normalized based on transaction importance and substitutability, scaling from 0.1 to 1.0.</li>
            </ul>
            
            <p style="margin-top: 8px; margin-bottom: 4px; font-weight: 600; color: #222;">Key Assumptions:</p>
            <ol style="margin-top: 0px; padding-left: 20px;">
                <li><b>No Substitutability</b>: In the short run, firms cannot easily substitute specialized suppliers (e.g., EUV Lithography).</li>
                <li><b>Linear Transmission</b>: Stress propagates as a linear fraction of dependency strength.</li>
                <li><b>Static Topology</b>: The supply network structure remains fixed during the propagation period.</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<h3 style='font-size: 14.5px; font-weight: 600; margin-top: 1.5rem; margin-bottom: 0.5rem; font-family: -apple-system, BlinkMacSystemFont, sans-serif; color: #111;'>3. Deterministic Model & Propagation</h3>", unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size: 12.5px; line-height: 1.6; font-family: -apple-system, BlinkMacSystemFont, sans-serif; color: #333;">
            The propagation engine is a deterministic, iterative network model that simulates the transmission of distress.
            
            <p style="margin-top: 12px; margin-bottom: 4px; font-weight: 600; color: #222;">Mathematical Formulation:</p>
            Let $S_i(t)$ be the stress level of firm $i$ at iteration step $t$. When a shock is introduced, the initial stress rises. In each subsequent step, stress propagates along the network links according to:
            </div>
            """, unsafe_allow_html=True)
            
            # Keep standard LaTeX st.markdown so Streamlit parses and formats math equations perfectly
            st.markdown("""
            $$S_j(t+1) = S_j(t) + \\alpha \\sum_{i \\in N(j)} w_{ij} \\cdot S_i(t)$$
            """)
            
            st.markdown("""
            <div style="font-size: 12.5px; line-height: 1.6; font-family: -apple-system, BlinkMacSystemFont, sans-serif; color: #333;">
            Where:
            <ul style="margin-top: 0px; padding-left: 20px; margin-bottom: 12px;">
                <li>$w_{ij}$ is the normalized dependency weight between firm $i$ and firm $j$.</li>
                <li>$\\alpha$ is the decay/damping factor (controlling the propagation rate and preventing infinite escalation).</li>
                <li>$N(j)$ represents the direct neighbors (suppliers/customers) of firm $j$.</li>
            </ul>
            
            <p style="margin-top: 8px; margin-bottom: 4px; font-weight: 600; color: #222;">Key Simplifications:</p>
            <ul style="margin-top: 0px; padding-left: 20px;">
                <li>The model does not simulate dynamic market pricing or inventory stockpiles.</li>
                <li>It focuses strictly on the financial transmission mechanism (credit, liquidity, and operational solvency) over a short-to-medium time horizon.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<h3 style='font-size: 14.5px; font-weight: 600; margin-top: 1.5rem; margin-bottom: 0.5rem; font-family: -apple-system, BlinkMacSystemFont, sans-serif; color: #111;'>4. Dashboard Interpretation Guide</h3>", unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size: 12.5px; line-height: 1.6; font-family: -apple-system, BlinkMacSystemFont, sans-serif; color: #333;">
            The interactive network visualization provides an intuitive topological view of financial risk:
            
            <ul style="margin-top: 0px; padding-left: 20px;">
                <li><b>Nodes represent Firms</b>: Hovering over a node displays its name, country, value-chain segment, and stress metrics.</li>
                <li><b>Edges represent Supply Dependencies</b>: The thickness of the line represents the strength/importance of the relationship.</li>
                <li><b>Node Color (Stress Level)</b>:
                    <ul style="margin-top: 4px; padding-left: 20px; margin-bottom: 4px;">
                        <li><b>Green</b>: Healthy, low-stress ($<0.35$).</li>
                        <li><b>Yellow/Orange</b>: Moderate risk ($0.35 - 0.70$).</li>
                        <li><b>Red</b>: Highly distressed / Default-risk ($&gt;0.70$).</li>
                    </ul>
                </li>
                <li><b>Node Size (Systemic Importance)</b>: Nodes are scaled by their total degree centrality (total incoming and outgoing dependency weights), highlighting critical hubs.</li>
                <li><b>View Modes</b>:
                    <ul style="margin-top: 4px; padding-left: 20px;">
                        <li><b>Baseline</b>: The initial, unperturbed state of the network.</li>
                        <li><b>Impact ($\Delta$ Stress)</b>: Highlights the change in stress caused by the simulation, helping isolate the immediate propagation path.</li>
                        <li><b>Final</b>: The long-term steady-state stress after propagation.</li>
                    </ul>
                </li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<h3 style='font-size: 14.5px; font-weight: 600; margin-top: 1.5rem; margin-bottom: 0.5rem; font-family: -apple-system, BlinkMacSystemFont, sans-serif; color: #111;'>5. Key Insights & Observations</h3>", unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size: 12.5px; line-height: 1.6; font-family: -apple-system, BlinkMacSystemFont, sans-serif; color: #333;">
            Based on scenario testing, several key structural insights emerge:
            
            <ul style="margin-top: 0px; padding-left: 20px;">
                <li><b>Idiosyncratic Shock Transmission</b>: Shocks applied to highly centralized foundries (e.g., TSMC) cause widespread downstream distress across fabless designers and downstream device makers due to their massive structural importance.</li>
                <li><b>Sector-level Vulnerabilities</b>: Shocking specific manufacturing niches (e.g., Wafer Manufacturing or EUV Lithography) reveals critical single points of failure. Even small sectors can trigger systemic network defaults if they produce highly non-substitutable inputs.</li>
                <li><b>Systemic Tightening</b>: Broad macro liquidity shocks reveal that firms with high baseline leverage are highly fragile and quickly cross into distress territory even with minimal secondary propagation.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<h3 style='font-size: 14.5px; font-weight: 600; margin-top: 1.5rem; margin-bottom: 0.5rem; font-family: -apple-system, BlinkMacSystemFont, sans-serif; color: #111;'>6. Model Limitations</h3>", unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size: 12.5px; line-height: 1.6; font-family: -apple-system, BlinkMacSystemFont, sans-serif; color: #333;">
            While highly valuable for vulnerability mapping, the model has several design limitations:
            
            <ul style="margin-top: 0px; padding-left: 20px;">
                <li><b>Static Nature</b>: In reality, firms will attempt to adapt by sourcing new suppliers or raising capital. The model assumes a fixed, short-term structure with no substitution.</li>
                <li><b>Data Granularity</b>: Relationships are built on publicly documented B2B transactions. Private or proprietary contracts might not be fully captured.</li>
                <li><b>Macro Factors</b>: The propagation assumes constant macroeconomic variables (e.g., interest rates, inflation) unless explicitly introduced as part of a systemic shock scenario.</li>
            </ul>
            
            <blockquote style="font-size: 12.5px; line-height: 1.6; font-family: -apple-system, BlinkMacSystemFont, sans-serif; border-left: 3px solid #ccc; padding-left: 12px; margin-top: 15px; margin-bottom: 15px; color: #555; background-color: #fafafa; padding-top: 4px; padding-bottom: 4px;">
            <b>Disclaimer:</b> This dashboard is an academic scenario-analysis tool designed to explore network topology vulnerabilities. It does not constitute investment, credit, or financial advice.
            </blockquote>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br><br>", unsafe_allow_html=True)