# Financial Supply Chain Risk

This project analyses how financial stress can propagate through the semiconductor supply chain. It combines firm-level financial risk indicators with a weighted supply-chain network to simulate shock scenarios and identify systemic chokepoints.

The model is designed as a scenario-analysis tool rather than a forecasting model. Its purpose is to show how distress at one or more firms may spread through dependency links and affect other parts of the semiconductor value chain.

## Overview

Semiconductor production relies on a globally distributed network of specialised firms across design, fabrication, equipment, materials, chemicals, wafers, and packaging. Because many of these firms have limited substitutes, financial stress at one node can create secondary effects across the network.

This project addresses three questions:

- Which firms are most vulnerable to propagated financial stress?
- Which firms act as systemic chokepoints in the supply-chain network?
- How do different shock scenarios affect the distribution of stress across the industry?

## Methodology

### Financial Stress

Firm-level financial stress is measured using the Altman Z''-Score:

**Z'' = 6.56X1 + 3.26X2 + 6.72X3 + 1.05X4**

where:

- X1 = Working Capital / Total Assets
- X2 = Retained Earnings / Total Assets
- X3 = EBIT / Total Assets
- X4 = Book Equity / Total Liabilities

The Z''-Score is used because it relies on accounting variables that are generally available across jurisdictions and does not require market capitalisation data.

### Supply-Chain Network

The semiconductor supply chain is represented as a weighted directed graph:

- Nodes represent firms.
- Edges represent buyer-supplier or partner relationships.
- Edge weights represent dependency strength.
- Edge direction represents the assumed direction of stress transmission.

Buyer-supplier relationships transmit stress from supplier to buyer. Partner relationships are modelled as bidirectional links with reduced propagation strength.

### Shock Propagation

Shock scenarios apply an initial increase in stress to selected firms or groups of firms. Stress then propagates through the network according to edge direction, relationship weight, and the model's propagation parameters.

The main outputs are final firm-level stress, change in stress relative to baseline, firms above the distress threshold, systemic chokepoint rankings, and exposure by supply-chain segment or geography.

## Dashboard

The Streamlit dashboard contains three main sections:

**Network View**  
Visualises the supply-chain graph. Nodes represent firms, edges represent dependencies, and colours indicate stress levels after a shock scenario.

**Data Explorer**  
Summarises the analytical outputs of the model, including scenario metrics, largest stress increases, systemic chokepoints, and exposure by segment or geography.

**Explanation**  
Provides a short methodology guide and a link to download the full report.

## Data

The project uses public financial filings, SEC EDGAR XBRL data for US-listed firms, EDINET data for Japanese firms, annual reports for other international firms, and supply-chain relationships from various industry sources.

The final dataset includes semiconductor-related firms mapped to their respective positions in the value chain.

## Running the Project

Install dependencies:

`pip install -r requirements.txt`

Run the dashboard:

`streamlit run app.py`

## Limitations

The model simplifies real-world supply-chain dynamics. It does not account for inventories, contractual protections, dynamic substitution, government intervention, or real-time market responses. Edge weights are fixed during simulation and reflect available evidence combined with modelling judgement.

Results should be interpreted as relative indicators of systemic vulnerability, not as precise predictions of default or operational failure.