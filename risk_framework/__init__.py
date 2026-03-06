"""
risk_framework — Financial risk scoring and credit zone classification.

Inputs:  data/financials/{ticker}.parquet
Outputs: risk_framework/scores.csv

The scorer module computes Altman Z-Scores, supplemental financial ratios,
and assigns each firm-year to a credit zone (safe/grey/distress).
"""
