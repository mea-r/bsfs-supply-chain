# Financial Supply Chain Risk Platform

A production-ready platform for modeling how financial stress propagates through supply chains, built for fintech research teams managing trade credit and working-capital exposure.

## Quick Start (5 commands)

```bash
# 1. Install dependencies (uses uv)
uv sync

# 2. Seed demo data (no API keys required)
python main.py seed

# 3. OR fetch live data from SEC EDGAR + Yahoo Finance + FRED
python main.py all

# 4. Run tests
make test

# 5. Launch dashboard
python main.py dashboard
# or: streamlit run dashboard/app.py
```

Open http://localhost:8501 in your browser.

---

## System Architecture

```
bsfs-supply-chain/
├── config.yaml                    Central configuration (thresholds, firms, shock params)
├── main.py                        CLI pipeline orchestrator (seed/fetch/macro/scores/dashboard)
├── data_engineering/
│   ├── sec_scraper.py             SEC EDGAR XBRL + yfinance data collection
│   ├── ratio_calculator.py        Altman Z-Score + 5 supplemental ratios
│   ├── supply_chain_builder.py    Supply chain graph (edges.csv) from documented sources
│   ├── macro_fetcher.py           FRED macroeconomic time series (Fed Funds, CPI, IP, HY spread)
│   └── data_quality_report.py     Audit & flag data gaps → data_quality_report.md
├── risk_framework/
│   ├── scorer.py                  ScoreStore: in-memory risk score lookup
│   └── trade_credit.py            Trade credit exposure: PD from Z-Score, Expected Loss = AR × PD × LGD
├── propagation/
│   └── propagation_engine.py      Rule-based stress propagation (4 rules, 3 shocks)
├── dashboard/
│   ├── app.py                     Streamlit interactive dashboard (5 tabs)
│   └── graph_utils.py             Plotly network graph + macro chart builders
├── utils/
│   └── config.py                  Shared config loader (used by all modules)
├── scripts/
│   └── seed_demo_data.py          Generate realistic synthetic data (no API required)
├── tests/                         pytest suite — 61 tests, no external dependencies
├── data/
│   ├── financials/                {ticker}.parquet — normalized financial data
│   ├── supply_chain/edges.csv     Documented supply chain relationships
│   └── macro/macro_series.csv     Macro time series (FRED or bundled fallback)
└── risk_framework/scores.csv      Z-scores, ratios, credit zones for all firms/years
```

---

## Modules

### 1. Data Engineering (`data_engineering/`)

Collects, cleans, and normalizes financial data from SEC EDGAR (XBRL API), Yahoo Finance, and FRED.

- **sec_scraper.py**: Pulls US-GAAP XBRL facts for all configured firms. Falls back gracefully on API failures (logs and continues). Handles multiple XBRL tags for the same concept.
- **ratio_calculator.py**: Computes all 7 financial metrics with documented economic justification.
- **supply_chain_builder.py**: Builds `edges.csv` from documented OEM-supplier relationships (10-K disclosures). Inferred edges tagged `inferred_sector_structure` for auditability.
- **macro_fetcher.py**: Fetches FRED time series (Fed Funds Rate, CPI, Industrial Production, HY Credit Spread). Requires `FRED_API_KEY` env var; falls back to bundled approximate data for offline use.
- **data_quality_report.py**: Produces `data_quality_report.md` flagging missingness, anomalies, and Z-score coverage.

### 2. Financial Ratios & Risk (`risk_framework/`)

| Metric | Formula | Economic Purpose |
|--------|---------|-----------------|
| Altman Z-Score | 1.2·X1 + 1.4·X2 + 3.3·X3 + 0.6·X4 + 1.0·X5 | Composite bankruptcy predictor (Altman 1968) |
| Current Ratio | Current Assets / Current Liabilities | Short-term liquidity |
| Debt-to-Equity | Total Liabilities / Stockholders' Equity | Financial leverage |
| Interest Coverage | EBIT / Interest Expense | Debt service capacity |
| Days Payable Outstanding | (AP / COGS) × 365 | Supplier payment timing |
| Days Sales Outstanding | (AR / Revenue) × 365 | Customer collection speed |

**Credit zones:**
- 🟢 Safe: Z > 2.99
- 🟡 Grey: 1.81 < Z ≤ 2.99
- 🔴 Distress: Z ≤ 1.81

**Trade credit exposure (`trade_credit.py`):**
- PD estimated from Z-Score mapping (Safe ≈ 0.2%, Grey ≈ 2–15%, Distress ≈ 15–70%)
- Expected Loss = Accounts Receivable × PD × LGD (LGD = 60%, Moody's 2023 average)
- Portfolio-level summary of AR at risk and expected loss

### 3. Propagation Engine (`propagation/`)

Graph: nodes = firms, edges = supplier→buyer (direction of stress flow).

**4 Propagation Rules:**

| Rule | Trigger | Economic Basis |
|------|---------|----------------|
| R1: Direct Transmission | Supplier Z-score in distress zone | Buyer faces supply disruption risk |
| R2: Liquidity Cascade | Supplier Current Ratio < 1.0 | Illiquid supplier delays payments, strains buyer working capital |
| R3: Contagion Dampening | Per-hop attenuation (α=0.6) | Diversification, insurance, inventory buffers (Barrot & Sauvagnat 2016) |
| R4: Chokepoint Amplification | High out-degree + grey/distress zone | Suppliers serving many buyers have outsized systemic impact |

**3 Shock Scenarios:**

| ID | Name | Maps To | Parameter |
|----|------|---------|-----------|
| S1 | Interest Rate Spike | 2022 Fed hike cycle | Interest expense increase % |
| S2 | Demand Shock | COVID-19 shutdowns, chip shortage | OEM revenue reduction % |
| S3 | Key Supplier Failure | Bankruptcy, plant fire | Focal firm + severity scalar |

### 4. Dashboard (`dashboard/`)

Five tabs:

- **Network View**: Interactive Plotly graph — nodes colored by credit zone, **edge colors change with stress** (green→yellow→red after shock), edge width = relationship weight, stress paths highlighted after S3.
- **Firm Detail**: Z-score time series, ratio bar charts, upstream/downstream exposure table, before/after shock impact.
- **Risk Summary**: Chokepoint ranking table, stress propagation heatmap, before/after zone distribution, FRED macro time series chart.
- **Case Studies**: Three pre-built scenario narratives (BorgWarner bankruptcy, COVID demand shock, 2022 rate hike cycle) with auto-run simulation and before/after firm tables.
- **Trade Credit Exposure**: Firm-level AR at risk, PD estimates, expected loss; portfolio-level summary metrics.

---

## Configuration

All business-logic parameters live in `config.yaml`. No hardcoded values in business logic.

**To change sector:** Edit the `firms` section in `config.yaml` and replace `supply_chain_builder.py`'s `AUTOMOTIVE_EDGES` list with sector-appropriate relationships.

**Key parameters:**
```yaml
propagation:
  contagion_damping_alpha: 0.6         # Stress attenuation per hop
  chokepoint_outdegree_threshold: 3    # Min buyers for chokepoint classification
  chokepoint_amplification_factor: 1.4
  s1_ebit_passthrough: 0.5             # Fraction of interest spike hitting EBIT
  s2_operating_leverage: 1.5           # EBIT falls faster than revenue (fixed costs)
  max_propagation_hops: 5

ratios:
  credit_zones:
    safe_threshold: 2.99
    grey_threshold: 1.81
```

---

## Testing

```bash
make test
# or: python -m pytest tests/ -v
```

61 tests across 4 test modules:
- `test_ratio_calculator.py`: Formula correctness, zone boundaries, missing data handling
- `test_propagation_engine.py`: Shock scenarios, propagation direction, chokepoint detection
- `test_supply_chain_builder.py`: Edge schema, weight bounds, documented sources
- `test_data_quality_report.py`: Audit accuracy, report generation

All tests run with no external dependencies.

---

## Data Sources

| Type | Source | Access |
|------|--------|--------|
| Financial statements | SEC EDGAR XBRL API | Free, no key required |
| Market data | Yahoo Finance (yfinance) | Free |
| Supply chain edges | 10-K disclosures (documented) | Hardcoded with citations |
| Macro time series | FRED (Federal Reserve) | Free — set `FRED_API_KEY` env var; bundled fallback available |

---

## Economic Assumptions & Limitations

1. **Altman Z-Score for large firms**: The model was calibrated on smaller US manufacturers. Large OEMs (Toyota, Ford, GM) will tend to score lower than their true credit quality due to high absolute liabilities. Use Z-Score changes directionally, not as absolute credit metrics for these firms.

2. **Operating leverage (S2)**: Fixed at 1.5× for demand shocks (configurable in `config.yaml`). Real operating leverage varies by firm cost structure.

3. **Edge weights**: Tier-2→Tier-1 weights are estimated from disclosed revenue concentration. Tier-3→Tier-2 weights where not directly disclosed are tagged `inferred_sector_structure`.

4. **Propagation dampening α=0.6**: Based on Barrot & Sauvagnat (2016) supply chain propagation estimates. May need recalibration for other sectors.

5. **Trade credit PD mapping**: Piecewise linear approximation calibrated to Altman's original study and Moody's 2023 default data. Not a formally calibrated PD model.

---

## Makefile Commands

```bash
make install        # Install Python dependencies (uv sync)
make supply_chain   # Build edges.csv (no API required)
make data           # Fetch live financial data (SEC EDGAR + yfinance)
make macro          # Fetch macroeconomic data from FRED
make scores         # Compute ratios and Z-scores
make quality        # Generate data_quality_report.md
make dashboard      # Launch Streamlit dashboard
make test           # Run pytest suite
make all            # Full pipeline end-to-end (data + macro + scores + quality)
make seed           # Seed realistic demo data (no API)
make clean          # Remove generated data files
```

Alternatively, use the CLI:
```bash
python main.py seed       # Seed demo data
python main.py fetch      # Fetch financial data
python main.py macro      # Fetch macro data
python main.py scores     # Compute ratios
python main.py dashboard  # Launch dashboard
python main.py all        # Run full pipeline
```
