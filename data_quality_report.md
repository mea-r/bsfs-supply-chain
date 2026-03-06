# Data Quality Report

Generated: 2026-03-06 18:42:43

Sector: **automotive**

Firms: 11  |  Firm-years: 53  |  Years: [2019, 2020, 2021, 2022, 2023]

---

## 1. Field Completeness

| Field | Missing % |
|-------|-----------|
| total_assets | 0.0% |
| current_assets | 0.0% |
| current_liabilities | 0.0% |
| retained_earnings | 0.0% |
| ebit | 0.0% |
| market_cap | 0.0% |
| revenue | 0.0% |
| total_liabilities | 0.0% |
| interest_expense | 0.0% |
| accounts_payable | 0.0% |
| accounts_receivable | 0.0% |
| cogs | 0.0% |
| long_term_debt | 0.0% |
| stockholders_equity | 0.0% |

---

## 2. Firm-Level Completeness

| Ticker | Missing % | Status |
|--------|-----------|--------|
| ADNT | 0.0% | 🟢 OK |
| APTV | 0.0% | 🟢 OK |
| BWA | 0.0% | 🟢 OK |
| DAN | 0.0% | 🟢 OK |
| F | 0.0% | 🟢 OK |
| GM | 0.0% | 🟢 OK |
| LEA | 0.0% | 🟢 OK |
| MGA | 0.0% | 🟢 OK |
| MOD | 0.0% | 🟢 OK |
| STLA | 0.0% | 🟢 OK |
| TM | 0.0% | 🟢 OK |

---

## 3. Anomaly Flags

No anomalies detected.

---

## 4. Z-Score Coverage

- Computed: **53** of 53 firm-years
- Missing: **0.0%**

Zone distribution: {'distress': 21, 'grey': 20, 'safe': 12}

---

## 5. Methodology Notes

- All financial data sourced from SEC EDGAR XBRL API (10-K annual filings).
- Market cap sourced from yfinance (year-end close × shares outstanding).
- Supply chain edges sourced from documented 10-K disclosures; inferred edges tagged `inferred_sector_structure`.
- Missing values are **never silently imputed**. Z-scores with >2 missing components return NaN.
- All transformations are code-reproducible (no manual steps).