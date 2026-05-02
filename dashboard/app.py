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

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Semiconductor Supply Chain Risk",
    page_icon="🔌",
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
    return pd.read_csv(path)

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

# ------------------------------------------------------------------
# UI: Sidebar Settings & Filters
# ------------------------------------------------------------------
st.sidebar.title("⚙️ Dashboard Settings")

# --- Visual Settings ---
with st.sidebar.expander("💡 Visual Settings", expanded=True):
    show_labels = st.checkbox("Show Labels on Graph", value=True, help="Display company names permanently on the graph.")
    show_legend = st.checkbox("Show Risk Legend", value=True)
    physics_engine = st.selectbox(
        "Physics Engine", 
        ["Barnes-Hut (Constellation)", "ForceAtlas2 (Clustered)", "Repulsion (Spread out)"], 
        index=1  # Default to ForceAtlas2
    )
    layout_stiffness = st.slider("Node Separation / Spread", 0.5, 5.0, 1.0) # Default to 1.0

# --- Filters ---
st.sidebar.title("🔍 Filters")
nodes_df = load_nodes()
edges_df = load_edges()

# Value Chain Filter
with st.sidebar.expander("🏢 Value Chain Category", expanded=False):
    categories = sorted(nodes_df["Value Chain Category"].dropna().unique().tolist())
    cat_mode = st.radio("Category Filter Mode", ["All", "Custom"], index=0, key="cat_mode", horizontal=True)
    if cat_mode == "Custom":
        selected_categories = st.multiselect("Select Categories", categories, default=categories)
    else:
        selected_categories = categories

# Country Filter
with st.sidebar.expander("🌍 Country", expanded=False):
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
            stress=row["Stress (logistic)"]
        )

    for _, row in edges_df.iterrows():
        try:
            u = str(int(float(row["a_id"])))
            v = str(int(float(row["b_id"])))
        except (ValueError, TypeError):
            continue

        if u in valid_ids and v in valid_ids:
            if row.get("supplier"): G.add_edge(u, v, strength=row["relationship_strength"])
            if row.get("customer"): G.add_edge(v, u, strength=row["relationship_strength"])

    return G

G = build_network(filtered_nodes, edges_df)

# ------------------------------------------------------------------
# Layout & Tabs
# ------------------------------------------------------------------
st.title("🔌 Semiconductor Supply Chain Risk Dashboard")

tab1, tab2 = st.tabs(["🌐 Network View", "📊 Data Explorer"])

# ==================================================================
# TAB 1: Network View
# ==================================================================
with tab1:
    if show_legend:
        # High-visibility legend
        st.info("💡 **Risk Zones (Z'' Score):** &nbsp;&nbsp; 🟢 Safe (>2.6) &nbsp;&nbsp; | &nbsp;&nbsp; 🟡 Grey (1.1-2.6) &nbsp;&nbsp; | &nbsp;&nbsp; 🔴 Distress (≤1.1) &nbsp;&nbsp; | &nbsp;&nbsp; ⚫ Unknown")

    if len(G.nodes) == 0:
        st.warning("No nodes match the selected filters.")
    else:
        # Build Pyvis Network (directed=False to remove arrows)
        net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="#333", directed=False)
        
        # Configure physics for an "Obsidian-style" interactive constellation
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
            color = get_zone_color(attrs['z_score'])
            # HTML for hover
            title = f"<b>{node}</b><br>Category: {attrs['category']}<br>Z'' Score: {attrs['z_score']:.2f}"
            net.add_node(
                str(node),
                label=attrs.get("name", str(node)) if show_labels else " ",               title=title,
                color=color, 
                size=25,
                borderWidth=0,
                font={"size": 16, "color": "#333"}
            )
            
        # Add edges
        for u, v, data in G.edges(data=True):
            strength = data.get('strength', 1)

            u_clean = str(u).replace(".0", "")
            v_clean = str(v).replace(".0", "")

            if u_clean in net.get_nodes() and v_clean in net.get_nodes():
                # Exponential scaling to create a large spread between widths (1->1, 2->2.5, 3->5, 4->8.5, 5->13)
                width_map = {1: 1, 2: 2.5, 3: 5.0, 4: 8.5, 5: 13.0}
                width = width_map.get(int(strength), strength * 2)

                # Physics: stronger ties mean shorter spring length (they pull closer)
                spring_len = max(30, 200 - (strength * 30))

                net.add_edge(
                    u, v,
                    value=width,
                    title=f"Strength: {strength}",
                    color={"color": "#b3b3b3", "highlight": "#555"},
                    length=spring_len
                )
            
        # Save and render interactively
        # Use a consistent path for the temporary file
        tmp_file = ROOT / "dashboard" / "pyvis_graph.html"
        net.save_graph(str(tmp_file))
        
        with open(tmp_file, 'r', encoding='utf-8') as f:
            source_code = f.read()
            
        components.html(source_code, height=710)

# ==================================================================
# TAB 2: Data Explorer
# ==================================================================
with tab2:
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
        st.subheader("🎯 Risk Zone Distribution")
        # Assign zones for the chart
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
        st.subheader("🏢 Financial Health by Category")
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
        st.subheader("🌍 Geographic Risk (Avg Z'')")
        geo_df = filtered_nodes.groupby("Country")["Z''"].mean().reset_index().sort_values("Z''")
        fig_geo = px.bar(
            geo_df, x="Z''", y="Country", orientation='h',
            color="Z''", color_continuous_scale="RdYlGn",
            title="Lower is more risky"
        )
        fig_geo.update_layout(height=350, yaxis_title=None)
        st.plotly_chart(fig_geo, use_container_width=True)

    with col4:
        st.subheader("🔥 Top 10 High-Stress Companies")
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
    st.subheader("🕸️ Supply Chain Hubs (Dependency Connectivity)")
    # Calculate degree from edges
    all_conns = pd.concat([edges_df["company_a"], edges_df["company_b"]])
    conn_counts = all_conns.value_counts().reset_index()
    conn_counts.columns = ["Company", "Connections"]
    # Filter to only companies in our filtered node list
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
    with st.expander("📄 Raw Company Data", expanded=False):
        st.dataframe(filtered_nodes.sort_values("Ranking"), use_container_width=True, hide_index=True)
    
    with st.expander("🔗 Raw Dependency Data", expanded=False):
        valid_companies = set(filtered_nodes["id"].tolist())
        display_edges = edges_df[
            (edges_df["company_a"].isin(valid_companies)) |
            (edges_df["company_b"].isin(valid_companies))
        ]
        st.dataframe(display_edges, use_container_width=True, hide_index=True)
