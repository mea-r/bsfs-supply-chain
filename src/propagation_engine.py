
#Financial Stress Propagation Engine — Semiconductor Supply Chain
#BSFS Financial Supply Chain Risk


#Output: console report + propagation_results.json

import json
import os
import sys
import warnings
from typing import Optional

import networkx as nx
import pandas as pd


# Global constants
ALPHA = 0.1 # change to ~0.3 for more realistic scenario
EPSILON = 1e-4 
MAX_STEPS = 50 

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

FIRMS_CSV   = os.path.join(PROJECT_ROOT, "data", "Final Table (csv).csv")
EDGES_CSV   = os.path.join(PROJECT_ROOT, "data", "Dependency relationships.csv")

OUTPUT_JSON = os.path.join(CURRENT_DIR, "propagation_results.json")



#some of the company names in the edges file don't match the firms file, that's why we need these aliases.
ALIASES: dict = {
    "Amkor":           "Amkor Technology",
    "Google":          "Alphabet (Google)",
    "Siemens":         "Siemens EDA (Mentor Graphics)",
    "Cadence":         "Cadence Design Systems",
    "Globalfoundries": "GlobalFoundries",
    "Micron":          "Micron Technology",
    "Axcelis":         "Axcelis Technologies",
}


# data loading


def load_firms(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()
    return df

def load_edges(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()
    for flag in ("supplier", "customer", "partner"):
        col = df[flag]
        if col.dtype == object:
            df[flag] = col.str.upper().map({"TRUE": True, "FALSE": False})
        else:
            df[flag] = col.astype(bool)
    return df

def resolve_name(raw: str, known: set) -> Optional[str]:

    if raw in known:
        return raw
    aliased = ALIASES.get(raw)
    if aliased and aliased in known:
        return aliased
    for name in known:
        if name.lower() == raw.lower():
            return name
    return None


# Graph construction

def normalize_weight(strength: float) -> float:
    """Linear map from relationship_strength ∈ [1,5] to edge weight ∈ [0.1, 1.0].
    Stronger documented relationships carry proportionally more stress."""
    return 0.1 + (strength - 1) * (0.9 / 4.0)


def build_graph(firms_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    known_names: set = set(firms_df["Company"])

    # Adding every firm as a node — even those with no dependency rows
    for _, row in firms_df.iterrows():
        G.add_node(
            row["Company"],
            z_score=float(row["Z''"]),
            stress_baseline=float(row["Stress (logistic)"]),
            value_chain_category=str(row["Value Chain Category"]),
            country=str(row["Country"]),
        )

    edge_candidates: dict = {}
    skipped: set = set()

    for _, row in edges_df.iterrows():
        raw_a, raw_b = str(row["company_a"]), str(row["company_b"])
        a = resolve_name(raw_a, known_names)
        b = resolve_name(raw_b, known_names)

        if a is None:
            if raw_a not in skipped:
                warnings.warn(f"[WARN] '{raw_a}' not in firms list — edges skipped.")
                skipped.add(raw_a)
            continue
        if b is None:
            if raw_b not in skipped:
                warnings.warn(f"[WARN] '{raw_b}' not in firms list — edges skipped.")
                skipped.add(raw_b)
            continue

        w = normalize_weight(float(row["relationship_strength"]))
        is_supplier = bool(row["supplier"])
        is_customer = bool(row["customer"])
        is_partner  = bool(row["partner"])

        def _register(src: str, dst: str, weight: float) -> None:
            if src == dst:
                return  
            key = (src, dst)
            edge_candidates[key] = max(edge_candidates.get(key, 0.0), weight)

        if is_supplier:
            _register(a, b, w)
        if is_customer:
            _register(b, a, w)
        if is_partner:
            _register(a, b, w / 2.0)
            _register(b, a, w / 2.0)


    for (src, dst), weight in edge_candidates.items():
        G.add_edge(src, dst, weight=weight)

    return G



def compute_centrality(G: nx.DiGraph) -> dict:

    return nx.betweenness_centrality(G, normalized=True, weight="weight")

def compute_hhi(G: nx.DiGraph) -> dict:
   # HHI_j = Σ (w_ij / Σw_kj)²
    hhi: dict = {}
    for j in G.nodes():
        in_weights = [G[i][j]["weight"] for i in G.predecessors(j)]
        if not in_weights:
            hhi[j] = 0.0
        else:
            total = sum(in_weights)
            hhi[j] = sum((w / total) ** 2 for w in in_weights)
    return hhi


def compute_vulnerability(G: nx.DiGraph, hhi: dict) -> dict:

    # v_j = 0.6 × s_j(baseline) + 0.4 × HHI_j
    return {
        j: 0.6 * G.nodes[j]["stress_baseline"] + 0.4 * hhi[j]
        for j in G.nodes()
    }



### Propagation engine ###

def propagate(
    G: nx.DiGraph,
    stress_init: dict,
    delta_init: dict,
    centrality: dict,
    vulnerability: dict,
    alpha: float = ALPHA,
) -> tuple:
    
    """
    our propagaration fornula as we decided on the meeting :
      Δs(t+1)_j = α × Σ_{i:(i,j)∈E} w_ij × (1 + c_i) × (1 + v_j) × Δs(t)_i
      s(t+1)_j  = min(1,  s(t)_j + Δs(t+1)_j)

    Stopping rule: max_j effective_Δs(t+1)_j < ε  OR  t ≥ MAX_STEPS
    """
    s = dict(stress_init)          
    delta = dict(delta_init)      

    delta_history: list = [max(delta.values()) if delta else 0.0]

    for _ in range(MAX_STEPS):
        new_delta: dict = {}

        for j in G.nodes():
            v_j = vulnerability[j]
            inflow = 0.0
            for i in G.predecessors(j):
                w_ij = G[i][j]["weight"] 
                c_i  = centrality[i]
                inflow += w_ij * (1.0 + c_i) * (1.0 + v_j) * delta.get(i, 0.0)
            new_delta[j] = alpha * inflow

    
        effective_delta: dict = {}
        for j in G.nodes():
            headroom = max(0.0, 1.0 - s[j])
            realised = min(new_delta[j], headroom)
            s[j] += realised
            effective_delta[j] = realised

        max_d = max(effective_delta.values()) if effective_delta else 0.0
        delta_history.append(max_d)
        delta = effective_delta

        if max_d < EPSILON:
            break

    return s, delta_history



# Scenario runner

def run_scenario(
    name: str,
    G: nx.DiGraph,
    shocked_firms: list,
    shock_delta: float,
    centrality: dict,
    vulnerability: dict,
    alpha: float = ALPHA,
) -> dict:
   
   #baseline 
    stress_init = {n: G.nodes[n]["stress_baseline"] for n in G.nodes()}
    delta_init  = {n: 0.0 for n in G.nodes()}

    for firm in shocked_firms:
        if firm not in G:
            warnings.warn(f"[WARN] Shocked firm '{firm}' not in graph — skipped.")
            continue

        headroom = max(0.0, 1.0 - stress_init[firm])
        effective_shock = min(shock_delta, headroom)
        stress_init[firm] += effective_shock
        delta_init[firm]   = effective_shock

    stress_final, delta_history = propagate(
        G, stress_init, delta_init, centrality, vulnerability, alpha=alpha
    )

    # Δ stress = change relative to baseline 
    stress_change = {
        n: stress_final[n] - G.nodes[n]["stress_baseline"] for n in G.nodes()
    }

    rounds = len(delta_history) - 1

    return {
        "name": name,
        "shocked_firms": shocked_firms,
        "stress_init": stress_init,
        "stress_final": stress_final,
        "stress_change": stress_change,
        "delta_history": delta_history,
        "rounds": rounds,
    }



def sanity_check(result: dict, G: nx.DiGraph) -> None:

    stress_final  = result["stress_final"]
    shocked_set   = set(result["shocked_firms"])
    delta_history = result["delta_history"]

    max_s = max(stress_final.values())
    ok1 = max_s <= 1.0 + 1e-9
    print(f"    [1] All stresses ≤ 1.0 ............. {'PASS' if ok1 else f'FAIL  (max={max_s:.8f})'}")

    ok2 = True
    for n in G.nodes():
        if G.in_degree(n) == 0 and n not in shocked_set:
            baseline = G.nodes[n]["stress_baseline"]
            if abs(stress_final[n] - baseline) > 1e-9:
                ok2 = False
                print(f"         ↳ FAIL: {n}  baseline={baseline:.4f}  final={stress_final[n]:.6f}")
    print(f"    [2] Unshocked source nodes unchanged  {'PASS' if ok2 else 'FAIL'}") 
    prop_rounds = delta_history[1:]
    ok3 = all(
        prop_rounds[i] >= prop_rounds[i + 1] - 1e-12
        for i in range(len(prop_rounds) - 1)
    )
    print(f"    [3] Max Δs decays monotonically ..... {'PASS' if ok3 else 'FAIL  (non-monotonic)'}")



#  prints


def print_scenario_results(
    result: dict,
    G: nx.DiGraph,
    centrality: dict,
    top_n: int = 10,
    top_choke: int = 5,
) -> None:
    W = 72
    print(f"\n{'═' * W}")
    print(f"  {result['name']}")
    print(f"{'═' * W}")
    print(f"  Shocked : {', '.join(result['shocked_firms']) if len(result['shocked_firms']) <= 6 else str(len(result['shocked_firms'])) + ' firms (all)'}")
    print(f"  Rounds  : {result['rounds']}")
    print()

    # top N most-affected firms
    ranked = sorted(G.nodes(), key=lambda n: result["stress_change"][n], reverse=True)
    print(f"  Top {top_n} Most Affected Firms (by Δ stress):")
    print(f"  {'─' * 70}")
    print(f"  {'Firm':<33} {'Baseline':>8} {'Final':>7} {'Δ Stress':>9}  Category")
    print(f"  {'─' * 70}")
    for firm in ranked[:top_n]:
        cat = G.nodes[firm]["value_chain_category"]
        cat_disp = (cat[:26] + "~") if len(cat) > 27 else cat
        baseline = G.nodes[firm]["stress_baseline"]
        final    = result["stress_final"][firm]
        delta    = result["stress_change"][firm]
        print(f"  {firm:<33} {baseline:>8.4f} {final:>7.4f} {delta:>9.4f}  {cat_disp}")

    print()

    # top chokepoints by betweenness centrality
    top_central = sorted(centrality, key=centrality.get, reverse=True)[:top_choke]
    print(f"  Top {top_choke} Chokepoints by Betweenness Centrality:")
    print(f"  {'─' * 46}")
    print(f"  {'Firm':<33} {'Centrality':>12}")
    print(f"  {'─' * 46}")
    for firm in top_central:
        print(f"  {firm:<33} {centrality[firm]:>12.6f}")

    print()
    print("  Sanity Checks:")
    sanity_check(result, G)
    print()


#  JSON exportation
def build_json(
    G: nx.DiGraph,
    centrality: dict,
    vulnerability: dict,
    scenarios: list,
) -> dict:

    network = {
        "nodes": [
            {
                "name":             n,
                "z_score":          float(G.nodes[n]["z_score"]),
                "stress_baseline":  float(G.nodes[n]["stress_baseline"]),
                "category":         G.nodes[n]["value_chain_category"],
                "country":          G.nodes[n]["country"],
            }
            for n in G.nodes()
        ],
        "edges": [
            {"source": u, "target": v, "weight": float(d["weight"])}
            for u, v, d in G.edges(data=True)
        ],
        "baseline_stress": {n: float(G.nodes[n]["stress_baseline"]) for n in G.nodes()},
        "centrality":      {n: float(v) for n, v in centrality.items()},
        "vulnerability":   {n: float(v) for n, v in vulnerability.items()},
    }

    out = {"network": network}
    for idx, res in enumerate(scenarios, start=1):
        out[f"scenario_{idx}"] = {
            "name":          res["name"],
            "shocked_firms": res["shocked_firms"],
            "stress_final":  {n: float(v) for n, v in res["stress_final"].items()},
            "stress_change": {n: float(v) for n, v in res["stress_change"].items()},
            "rounds":        res["rounds"],
        }
    return out


####  MAIN ###

def main() -> None:

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("Loading data ...")
    firms_df = load_firms(FIRMS_CSV)
    edges_df = load_edges(EDGES_CSV)
    print(f"  {len(firms_df)} firms  |  {len(edges_df)} dependency rows")

    # GRAPH BUILDING
    print("\nBuilding supply-chain graph ...")
    G = build_graph(firms_df, edges_df)
    print(f"  {G.number_of_nodes()} nodes  |  {G.number_of_edges()} directed edges")

    # METRIC COMPUTATION
    print("\nComputing centrality, HHI, and vulnerability ...")
    centrality    = compute_centrality(G)
    hhi           = compute_hhi(G)
    vulnerability = compute_vulnerability(G, hhi)

    top5 = sorted(centrality, key=centrality.get, reverse=True)[:5]
    print(f"  Top-5 betweenness: {', '.join(top5)}")

    # Scenario 1: Idiosyncratic — ASML distress 
    # ASML is the sole producer of EUV lithography machines; a severe shock
    print("\nScenario 1: ASML idiosyncratic distress (Ds = +0.50) ...")
    asml_nodes = [n for n in G.nodes() if n == "ASML"]
    if not asml_nodes:
        raise ValueError("ASML not found in graph — check firm name in CSV.")
    s1 = run_scenario(
        "Scenario 1 — Idiosyncratic: ASML Distress  (Ds = +0.50)",
        G, asml_nodes, 0.50, centrality, vulnerability,
    )

    # Scenario 2: Sector shock — Wafer Manufacturing 
    # A simultaneous supply shock to all wafer producers 
    print("Scenario 2: Wafer Manufacturing sector shock (Ds = +0.35) ...")
    wafer_firms = [
        n for n in G.nodes()
        if G.nodes[n]["value_chain_category"] == "Wafer Manufacturing"
    ]
    print(f"  Wafer firms identified: {wafer_firms}")
    s2 = run_scenario(
        "Scenario 2 — Sector Shock: Wafer Manufacturing  (Ds = +0.35)",
        G, wafer_firms, 0.35, centrality, vulnerability,
    )

    # Scenario 3: Systemic — credit tightening
    # A macro liquidity shock raises funding costs for every firm simultaneously

    print("Scenario 3: Credit tightening — all firms (Ds = +0.15) ...")
    all_firms = list(G.nodes())
    s3 = run_scenario(
        "Scenario 3 — Systemic: Credit Tightening  (Ds = +0.15)",
        G, all_firms, 0.15, centrality, vulnerability,
    )

    # PRINT RESULTS
    for res in (s1, s2, s3):
        print_scenario_results(res, G, centrality)

    # JSON EXPORT
    print(f"Exporting to {OUTPUT_JSON} ...")
    payload = build_json(G, centrality, vulnerability, [s1, s2, s3])
    with open(OUTPUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print("Done.")


if __name__ == "__main__":
    main()
