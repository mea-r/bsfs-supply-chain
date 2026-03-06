"""
data_engineering — Data collection, cleaning, and normalization module.

Pipeline:
  1. sec_scraper.py      — pulls XBRL financial statements from SEC EDGAR
  2. ratio_calculator.py — computes all financial ratios from raw statements
  3. supply_chain_builder.py — constructs edges.csv from config + sector knowledge
  4. data_quality_report.py  — produces data_quality_report.md
"""
