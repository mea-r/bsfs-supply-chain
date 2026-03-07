# Data Quality Report

Generated: 2026-03-07 01:54:06

Sector: **automotive**

Firms: 8  |  Firm-years: 40  |  Years: [2019, 2020, 2021, 2022, 2023]

---

## 1. Field Completeness

| Field | Missing % |
|-------|-----------|
| market_cap | 100.0% ⚠️ |
| long_term_debt | 67.5% ⚠️ |
| total_liabilities | 50.0% ⚠️ |
| accounts_payable | 37.5% ⚠️ |
| interest_expense | 32.5% ⚠️ |
| cogs | 30.0% |
| ebit | 25.0% |
| accounts_receivable | 25.0% |
| revenue | 20.0% |
| total_assets | 12.5% |
| current_assets | 12.5% |
| current_liabilities | 12.5% |
| retained_earnings | 12.5% |
| stockholders_equity | 12.5% |

---

## 2. Firm-Level Completeness

| Ticker | Missing % | Status |
|--------|-----------|--------|
| TM | 100.0% | 🔴 UNRELIABLE — >50% data missing |
| ADNT | 50.0% | 🟡 CAUTION — elevated missingness |
| GM | 21.4% | 🟢 OK |
| STLA | 21.4% | 🟢 OK |
| LEA | 20.0% | 🟢 OK |
| F | 18.6% | 🟢 OK |
| DAN | 14.3% | 🟢 OK |
| APTV | 11.4% | 🟢 OK |

---

## 3. Anomaly Flags

| Ticker | Year | Issue | Value |
|--------|------|-------|-------|
| TM | 2019 | Zero or missing revenue | nan |
| TM | 2020 | Zero or missing revenue | nan |
| TM | 2021 | Zero or missing revenue | nan |
| TM | 2022 | Zero or missing revenue | nan |
| TM | 2023 | Zero or missing revenue | nan |
| APTV | 2019 | Zero or missing revenue | nan |
| APTV | 2020 | Zero or missing revenue | nan |
| APTV | 2021 | Zero or missing revenue | nan |

---

## 4. Z-Score Coverage

- Computed: **35** of 40 firm-years
- Missing: **12.5%**

Zone distribution: {'distress': 30, 'grey': 5, 'unknown': 5}

---

## 5. Methodology Notes

- All financial data sourced from SEC EDGAR XBRL API (10-K annual filings).
- Market cap sourced from yfinance (year-end close × shares outstanding).
- Supply chain edges sourced from documented 10-K disclosures; inferred edges tagged `inferred_sector_structure`.
- Missing values are **never silently imputed**. Z-scores with >2 missing components return NaN.
- All transformations are code-reproducible (no manual steps).