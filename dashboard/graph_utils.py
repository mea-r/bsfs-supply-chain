"""
graph_utils.py — Network visualization helpers for the dashboard.

Builds Plotly network graphs and pyvis HTML exports from
PropagationEngine state dictionaries.
"""

import math
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


ZONE_COLORS = {
    "safe": "#2ecc71",
    "grey": "#f39c12",
    "distress": "#e74c3c",
    "unknown": "#95a5a6",
}


def build_plotly_graph(
    graph: nx.DiGraph,
    node_states: dict,
    config: dict,
    highlight_path: list = None,
) -> go.Figure:
    """
    Build a Plotly scatter+line network graph of the supply chain.

    Node color = credit zone (green/yellow/red).
    Node size = proportional to total assets (scaled).
    Edge thickness = relationship weight.
    Highlighted path nodes are outlined in bold.

    Parameters
    ----------
    graph : nx.DiGraph
        The supply chain NetworkX graph.
    node_states : dict
        Current node states from PropagationEngine.
    config : dict
        Loaded config.yaml.
    highlight_path : list, optional
        List of ticker strings to highlight (stress propagation path).

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    # Compute layout
    pos = nx.spring_layout(graph, seed=42, k=2.0)

    edge_traces = []
    for src, tgt, data in graph.edges(data=True):
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]
        w = data.get("weight", 0.5)
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=max(1, w * config["dashboard"]["edge_weight_scale"]),
                      color="#bdc3c7"),
            hoverinfo="none",
            showlegend=False,
        ))

    # Node trace
    node_x, node_y = [], []
    node_colors, node_sizes, node_texts, node_hovers = [], [], [], []
    highlight_set = set(highlight_path or [])

    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        state = node_states.get(node, {})
        zone = state.get("credit_zone", "unknown")
        node_colors.append(ZONE_COLORS.get(zone, "#95a5a6"))

        # Size by total assets (log scale)
        ta = state.get("total_assets", float("nan"))
        if not pd.isna(ta) and ta > 0:
            size = max(10, min(40, math.log10(ta / 1e8) * 10 + 15))
        else:
            size = 15
        node_sizes.append(size)

        name = state.get("name", node)
        z = state.get("z_score", float("nan"))
        z_str = f"{z:.2f}" if not pd.isna(z) else "N/A"
        cr = state.get("current_ratio", float("nan"))
        cr_str = f"{cr:.2f}" if not pd.isna(cr) else "N/A"
        stress = state.get("stress_score", 0.0)

        node_texts.append(node)
        node_hovers.append(
            f"<b>{name} ({node})</b><br>"
            f"Zone: <b>{zone.upper()}</b><br>"
            f"Z-Score: {z_str}<br>"
            f"Current Ratio: {cr_str}<br>"
            f"Stress Score: {stress:.3f}<br>"
            f"In-degree: {graph.in_degree(node)} | Out-degree: {graph.out_degree(node)}"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line=dict(
                color=["#2c3e50" if n in highlight_set else "white"
                       for n in graph.nodes()],
                width=[3 if n in highlight_set else 1 for n in graph.nodes()],
            ),
            opacity=0.9,
        ),
        text=node_texts,
        textposition="top center",
        hovertext=node_hovers,
        hoverinfo="text",
        showlegend=False,
    )

    # Legend annotations
    legend_traces = []
    for zone, color in ZONE_COLORS.items():
        legend_traces.append(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(color=color, size=12, symbol="circle"),
            name=zone.capitalize(),
            showlegend=True,
        ))

    fig = go.Figure(
        data=edge_traces + [node_trace] + legend_traces,
        layout=go.Layout(
            title="Supply Chain Risk Network",
            showlegend=True,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="#f8f9fa",
            paper_bgcolor="white",
            legend=dict(
                title="Credit Zone",
                yanchor="top", y=0.99, xanchor="left", x=0.01,
            ),
        )
    )
    return fig


def build_stress_heatmap(heatmap_df: pd.DataFrame) -> go.Figure:
    """
    Build a Plotly heatmap of stress propagation intensity.

    Parameters
    ----------
    heatmap_df : pd.DataFrame
        Stress matrix from PropagationEngine.get_propagation_heatmap().

    Returns
    -------
    go.Figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_df.values,
        x=heatmap_df.columns.tolist(),
        y=heatmap_df.index.tolist(),
        colorscale="RdYlGn_r",
        colorbar=dict(title="Stress Intensity"),
        hovertemplate="From: %{y}<br>To: %{x}<br>Stress: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title="Stress Propagation Heatmap (Supplier → Buyer)",
        xaxis_title="Buyer",
        yaxis_title="Supplier",
        margin=dict(l=80, r=20, t=60, b=80),
    )
    return fig


def build_z_score_timeseries(scores_df: pd.DataFrame, tickers: list) -> go.Figure:
    """
    Build a time series chart of Z-scores for selected firms.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Full scores dataframe (all years).
    tickers : list
        List of ticker strings to plot.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    # Add safe/grey/distress threshold bands
    years = sorted(scores_df["year"].unique())
    fig.add_hrect(y0=2.99, y1=10, fillcolor="#2ecc71", opacity=0.05,
                  annotation_text="Safe Zone", annotation_position="top right")
    fig.add_hrect(y0=1.81, y1=2.99, fillcolor="#f39c12", opacity=0.08,
                  annotation_text="Grey Zone", annotation_position="top right")
    fig.add_hrect(y0=-5, y1=1.81, fillcolor="#e74c3c", opacity=0.05,
                  annotation_text="Distress Zone", annotation_position="top right")
    fig.add_hline(y=2.99, line_dash="dash", line_color="#2ecc71", opacity=0.6)
    fig.add_hline(y=1.81, line_dash="dash", line_color="#e74c3c", opacity=0.6)

    for ticker in tickers:
        firm_data = scores_df[scores_df["ticker"] == ticker].sort_values("year")
        if firm_data.empty:
            continue
        name = firm_data["name"].iloc[0] if "name" in firm_data.columns else ticker
        fig.add_trace(go.Scatter(
            x=firm_data["year"],
            y=firm_data["z_score"],
            mode="lines+markers",
            name=f"{ticker} ({name})",
            line=dict(width=2),
            marker=dict(size=8),
            hovertemplate=f"{ticker}<br>Year: %{{x}}<br>Z-Score: %{{y:.2f}}<extra></extra>",
        ))

    fig.update_layout(
        title="Altman Z-Score Time Series",
        xaxis_title="Year",
        yaxis_title="Altman Z-Score",
        legend_title="Firm",
        hovermode="x unified",
        plot_bgcolor="#f8f9fa",
    )
    return fig


def build_ratio_comparison(scores_df: pd.DataFrame, year: int) -> go.Figure:
    """
    Build a bar chart comparing key ratios across firms for a given year.

    Parameters
    ----------
    scores_df : pd.DataFrame
    year : int

    Returns
    -------
    go.Figure
    """
    df_year = scores_df[scores_df["year"] == year].copy()
    if df_year.empty:
        return go.Figure()

    df_year = df_year.dropna(subset=["z_score"])
    df_year = df_year.sort_values("z_score")

    colors = [ZONE_COLORS.get(z, "#95a5a6") for z in df_year["credit_zone"]]

    fig = go.Figure(go.Bar(
        x=df_year["ticker"],
        y=df_year["z_score"],
        marker_color=colors,
        text=[f"{z:.2f}" for z in df_year["z_score"]],
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>Z-Score: %{y:.2f}<br>"
            "<extra></extra>"
        ),
    ))
    fig.add_hline(y=2.99, line_dash="dash", line_color="#2ecc71",
                  annotation_text="Safe threshold")
    fig.add_hline(y=1.81, line_dash="dash", line_color="#e74c3c",
                  annotation_text="Distress threshold")
    fig.update_layout(
        title=f"Altman Z-Score by Firm ({year})",
        xaxis_title="Firm",
        yaxis_title="Z-Score",
        plot_bgcolor="#f8f9fa",
        showlegend=False,
    )
    return fig
