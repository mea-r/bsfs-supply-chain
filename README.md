# Semiconductor Supply Chain Risk Dashboard

An interactive platform for visualizing semiconductor supply chain dependencies and financial risk, built with Streamlit and Pyvis.

## Quick Start

```bash
# 1. Install dependencies (uses uv)
uv sync

# 2. Launch dashboard
streamlit run dashboard/app.py
```

Open http://localhost:8501 in your browser.

---

## Features

### 1. Interactive Supply Chain Network
- **Dynamic Constellation**: Real-time force-directed graph using the Pyvis engine.
- **Physics-Driven Correlation**: Stronger relationship links pull nodes closer together in space.
- **Risk Color-Coding**: Nodes are colored based on their financial health (Altman Z'' Score):
  - 🟢 **Safe** (Z'' > 2.6)
  - 🟡 **Grey** (1.1 < Z'' ≤ 2.6)
  - 🔴 **Distress** (Z'' ≤ 1.1)
  - ⚫ **Unknown**
- **Exponential Edge Weighting**: Tougher links are visually emphasized with significant thickness increases.
- **Customizable Layouts**: Toggle between Force-Directed (Obsidian-style), Circular, and Kamada-Kawai structures.

### 2. Data Explorer & Insights
- **Metric Dashboard**: Instant overview of ecosystem health, average risk scores, and distress levels.
- **Risk Zone Composition**: Donut chart showing the total distribution of safe vs. distressed firms.
- **Category Health Analysis**: Box plots comparing Z'' score stability across value chain categories.
- **Geographic Risk Heatmap**: Identification of regional vulnerabilities.
- **High-Stress Leaderboard**: Top 10 immediate "watch list" companies based on logistic stress indices.
- **Supply Chain Hubs**: Identification of the most connected "hub" firms in the semiconductor ecosystem.

---

## Data Sources

The dashboard utilizes two primary datasets located in the `data/` directory:
1. **`Final Table (csv).csv`**: Contains company financial metrics, including Z'' scores, stress indices, and value chain classifications.
2. **`Dependency relationships.csv`**: Contains documented supplier-customer relationships and link strengths.

---

## Technical Stack

- **Frontend**: Streamlit
- **Graph Engine**: Pyvis (Interactive Physics) & NetworkX
- **Data Processing**: Pandas
- **Visualization**: Plotly Express
