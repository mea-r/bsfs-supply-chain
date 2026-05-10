import networkx as nx
from src.propagation_engine import run_scenario

def run_idiosyncratic_shock(G: nx.DiGraph, centrality: dict, vulnerability: dict, target_firm: str, shock_delta: float, alpha: float) -> dict:
    """
    Apply a shock to a single specific firm.
    """
    if target_firm not in G.nodes():
        return None
    return run_scenario(
        f"Idiosyncratic Shock: {target_firm}",
        G,
        [target_firm],
        shock_delta,
        centrality,
        vulnerability,
        alpha=alpha
    )

def run_sector_shock(G: nx.DiGraph, centrality: dict, vulnerability: dict, target_sector: str, shock_delta: float, alpha: float) -> dict:
    """
    Apply a shock to all firms within a specific Value Chain Category.
    """
    sector_firms = [
        n for n in G.nodes()
        if G.nodes[n].get("value_chain_category") == target_sector or G.nodes[n].get("category") == target_sector
    ]
    if not sector_firms:
        return None
    return run_scenario(
        f"Sector Shock: {target_sector}",
        G,
        sector_firms,
        shock_delta,
        centrality,
        vulnerability,
        alpha=alpha
    )

def run_systemic_shock(G: nx.DiGraph, centrality: dict, vulnerability: dict, shock_delta: float, alpha: float) -> dict:
    """
    Apply a broad shock to all firms in the network simultaneously.
    """
    all_firms = list(G.nodes())
    return run_scenario(
        "Systemic Shock: All Firms",
        G,
        all_firms,
        shock_delta,
        centrality,
        vulnerability,
        alpha=alpha
    )
