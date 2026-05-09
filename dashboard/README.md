# Semiconductor Supply Chain Dashboard

This feature provides an interactive visualization layer for the semiconductor supply chain data. It is designed to integrate seamlessly with the existing `data/` and `data_engineering/` structures.

## Installation

Ensure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

## Running the Dashboard

Launch the Streamlit application:

```bash
streamlit run dashboard/app.py
```

## Features

### 1. Interactive Supply Chain Network
- **Force-Directed Graph**: Uses the Pyvis engine to create a dynamic, draggable constellation of semiconductor firms.
- **Risk Color-Coding**: 
  - 🟢 **Safe** (Z'' > 2.6)
  - 🟡 **Grey** (1.1 < Z'' ≤ 2.6)
  - 🔴 **Distress** (Z'' ≤ 1.1)
  - ⚫ **Unknown**
- **Relationship Strength**: Edge thicknesses are scaled exponentially to highlight critical supply chain dependencies.
- **Physics-Based Layouts**: Support for **ForceAtlas2**, **Barnes-Hut**, and **Repulsion** engines to optimize spatial separation and minimize edge crossing.

### 2. Analytical Data Explorer
- **System Metrics**: High-level summary of total companies, average risk scores, and percentage of firms in distress.
- **Sector Analysis**: Box plots and donut charts showing the distribution of financial health across different value chain categories.
- **Regional Risk**: Geographic bar charts identifying country-level concentrations of financial stress.
- **Hub Detection**: Visualization of the most connected companies (hubs) in the dependency network.

## Integration

The dashboard directly consumes:
- `data/Final Table (csv).csv`: Firm-level financial metrics and risk scores.
- `data/Dependency relationships.csv`: Supply chain linkage data.
