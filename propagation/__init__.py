"""
propagation — Supply chain financial stress propagation engine.

The PropagationEngine models how financial distress spreads through
the supplier network using configurable rule-based logic.

Inputs:
  - risk_framework/scores.csv (firm-year financial risk scores)
  - data/supply_chain/edges.csv (supply chain graph edges)
  - config.yaml (propagation parameters, shock definitions)

Outputs:
  - Updated node stress states after shock application
  - Stress propagation paths
  - Chokepoint analysis
"""
