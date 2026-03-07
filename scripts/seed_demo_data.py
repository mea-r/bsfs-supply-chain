"""
seed_demo_data.py — Generate realistic synthetic financial data for demo/offline use.

This script creates realistic (but synthetic) financial data for the automotive
supply chain firms, so the dashboard can be used without API access.

The data is calibrated to approximate real-world magnitudes for these firms
(e.g., Ford ~$176B revenue, Toyota ~$274B revenue in 2023) while being clearly
synthetic (values are rounded and approximate).

Run: python scripts/seed_demo_data.py
"""

import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import yaml

# Load config
with open(ROOT / "config.yaml") as f:
    config = yaml.safe_load(f)


# -----------------------------------------------------------------------
# Synthetic financial data
# Data calibrated to approximate real-world scale (in USD millions).
# Sources for scale estimates:
#   - Toyota FY2023 Annual Report
#   - Ford 10-K 2023
#   - GM 10-K 2023
#   - Stellantis 2023 Annual Report
#   - Magna, Aptiv, BorgWarner, Lear 10-K 2022/2023
#   - Adient, Dana, Modine 10-K 2022
# -----------------------------------------------------------------------
# Format: (ticker, name, year): {financial fields in USD actual, not millions}
SEED_DATA = [
    # ---- Toyota (TM) ----
    # ~$274B revenue, strong balance sheet, Z-score historically ~3.5-4.0
    {"ticker": "TM",  "name": "Toyota Motor Corporation", "year": 2019,
     "total_assets": 495_000_000_000, "current_assets": 145_000_000_000,
     "current_liabilities": 100_000_000_000, "total_liabilities": 355_000_000_000,
     "retained_earnings": 130_000_000_000, "ebit": 22_000_000_000,
     "market_cap": 190_000_000_000, "revenue": 275_000_000_000,
     "interest_expense": 2_500_000_000, "accounts_payable": 22_000_000_000,
     "accounts_receivable": 18_000_000_000, "cogs": 230_000_000_000,
     "long_term_debt": 80_000_000_000, "stockholders_equity": 140_000_000_000},
    {"ticker": "TM",  "name": "Toyota Motor Corporation", "year": 2020,
     "total_assets": 480_000_000_000, "current_assets": 138_000_000_000,
     "current_liabilities": 105_000_000_000, "total_liabilities": 345_000_000_000,
     "retained_earnings": 125_000_000_000, "ebit": 14_000_000_000,  # COVID impact
     "market_cap": 195_000_000_000, "revenue": 255_000_000_000,
     "interest_expense": 2_200_000_000, "accounts_payable": 20_000_000_000,
     "accounts_receivable": 17_000_000_000, "cogs": 215_000_000_000,
     "long_term_debt": 85_000_000_000, "stockholders_equity": 135_000_000_000},
    {"ticker": "TM",  "name": "Toyota Motor Corporation", "year": 2021,
     "total_assets": 510_000_000_000, "current_assets": 150_000_000_000,
     "current_liabilities": 108_000_000_000, "total_liabilities": 365_000_000_000,
     "retained_earnings": 135_000_000_000, "ebit": 24_000_000_000,
     "market_cap": 240_000_000_000, "revenue": 279_000_000_000,
     "interest_expense": 2_300_000_000, "accounts_payable": 23_000_000_000,
     "accounts_receivable": 19_000_000_000, "cogs": 237_000_000_000,
     "long_term_debt": 82_000_000_000, "stockholders_equity": 145_000_000_000},
    {"ticker": "TM",  "name": "Toyota Motor Corporation", "year": 2022,
     "total_assets": 525_000_000_000, "current_assets": 148_000_000_000,
     "current_liabilities": 110_000_000_000, "total_liabilities": 375_000_000_000,
     "retained_earnings": 138_000_000_000, "ebit": 20_000_000_000,  # supply disruption
     "market_cap": 205_000_000_000, "revenue": 272_000_000_000,
     "interest_expense": 3_000_000_000, "accounts_payable": 22_000_000_000,
     "accounts_receivable": 18_500_000_000, "cogs": 232_000_000_000,
     "long_term_debt": 84_000_000_000, "stockholders_equity": 150_000_000_000},
    {"ticker": "TM",  "name": "Toyota Motor Corporation", "year": 2023,
     "total_assets": 550_000_000_000, "current_assets": 155_000_000_000,
     "current_liabilities": 112_000_000_000, "total_liabilities": 385_000_000_000,
     "retained_earnings": 145_000_000_000, "ebit": 28_000_000_000,
     "market_cap": 230_000_000_000, "revenue": 305_000_000_000,
     "interest_expense": 3_500_000_000, "accounts_payable": 25_000_000_000,
     "accounts_receivable": 20_000_000_000, "cogs": 258_000_000_000,
     "long_term_debt": 86_000_000_000, "stockholders_equity": 165_000_000_000},

    # ---- Ford (F) ----
    # ~$176B revenue, moderate leverage, Z-score historically 1.8-2.5 (grey zone often)
    {"ticker": "F",   "name": "Ford Motor Company", "year": 2019,
     "total_assets": 257_000_000_000, "current_assets": 73_000_000_000,
     "current_liabilities": 65_000_000_000, "total_liabilities": 225_000_000_000,
     "retained_earnings": 15_000_000_000, "ebit": 4_500_000_000,
     "market_cap": 36_000_000_000, "revenue": 156_000_000_000,
     "interest_expense": 2_900_000_000, "accounts_payable": 22_000_000_000,
     "accounts_receivable": 14_000_000_000, "cogs": 140_000_000_000,
     "long_term_debt": 96_000_000_000, "stockholders_equity": 32_000_000_000},
    {"ticker": "F",   "name": "Ford Motor Company", "year": 2020,
     "total_assets": 267_000_000_000, "current_assets": 74_000_000_000,
     "current_liabilities": 67_000_000_000, "total_liabilities": 235_000_000_000,
     "retained_earnings": 10_000_000_000, "ebit": -2_000_000_000,  # COVID loss
     "market_cap": 38_000_000_000, "revenue": 127_000_000_000,
     "interest_expense": 2_800_000_000, "accounts_payable": 18_000_000_000,
     "accounts_receivable": 12_000_000_000, "cogs": 116_000_000_000,
     "long_term_debt": 100_000_000_000, "stockholders_equity": 32_000_000_000},
    {"ticker": "F",   "name": "Ford Motor Company", "year": 2021,
     "total_assets": 257_000_000_000, "current_assets": 78_000_000_000,
     "current_liabilities": 63_000_000_000, "total_liabilities": 220_000_000_000,
     "retained_earnings": 18_000_000_000, "ebit": 7_000_000_000,
     "market_cap": 80_000_000_000, "revenue": 136_000_000_000,
     "interest_expense": 2_700_000_000, "accounts_payable": 21_000_000_000,
     "accounts_receivable": 14_000_000_000, "cogs": 122_000_000_000,
     "long_term_debt": 96_000_000_000, "stockholders_equity": 37_000_000_000},
    {"ticker": "F",   "name": "Ford Motor Company", "year": 2022,
     "total_assets": 255_000_000_000, "current_assets": 79_000_000_000,
     "current_liabilities": 65_000_000_000, "total_liabilities": 218_000_000_000,
     "retained_earnings": 16_000_000_000, "ebit": 5_500_000_000,
     "market_cap": 55_000_000_000, "revenue": 158_000_000_000,
     "interest_expense": 3_400_000_000, "accounts_payable": 22_500_000_000,
     "accounts_receivable": 14_500_000_000, "cogs": 143_000_000_000,
     "long_term_debt": 94_000_000_000, "stockholders_equity": 37_000_000_000},
    {"ticker": "F",   "name": "Ford Motor Company", "year": 2023,
     "total_assets": 270_000_000_000, "current_assets": 82_000_000_000,
     "current_liabilities": 67_000_000_000, "total_liabilities": 228_000_000_000,
     "retained_earnings": 20_000_000_000, "ebit": 8_000_000_000,
     "market_cap": 50_000_000_000, "revenue": 176_000_000_000,
     "interest_expense": 3_800_000_000, "accounts_payable": 24_000_000_000,
     "accounts_receivable": 15_000_000_000, "cogs": 158_000_000_000,
     "long_term_debt": 97_000_000_000, "stockholders_equity": 42_000_000_000},

    # ---- General Motors (GM) ----
    {"ticker": "GM",  "name": "General Motors", "year": 2019,
     "total_assets": 228_000_000_000, "current_assets": 69_000_000_000,
     "current_liabilities": 67_000_000_000, "total_liabilities": 189_000_000_000,
     "retained_earnings": 26_000_000_000, "ebit": 7_300_000_000,
     "market_cap": 52_000_000_000, "revenue": 137_000_000_000,
     "interest_expense": 1_600_000_000, "accounts_payable": 22_000_000_000,
     "accounts_receivable": 12_000_000_000, "cogs": 120_000_000_000,
     "long_term_debt": 64_000_000_000, "stockholders_equity": 39_000_000_000},
    {"ticker": "GM",  "name": "General Motors", "year": 2020,
     "total_assets": 235_000_000_000, "current_assets": 72_000_000_000,
     "current_liabilities": 70_000_000_000, "total_liabilities": 197_000_000_000,
     "retained_earnings": 22_000_000_000, "ebit": 3_800_000_000,
     "market_cap": 75_000_000_000, "revenue": 122_000_000_000,
     "interest_expense": 1_900_000_000, "accounts_payable": 18_000_000_000,
     "accounts_receivable": 10_000_000_000, "cogs": 108_000_000_000,
     "long_term_debt": 70_000_000_000, "stockholders_equity": 38_000_000_000},
    {"ticker": "GM",  "name": "General Motors", "year": 2021,
     "total_assets": 244_000_000_000, "current_assets": 74_000_000_000,
     "current_liabilities": 70_000_000_000, "total_liabilities": 200_000_000_000,
     "retained_earnings": 27_000_000_000, "ebit": 8_500_000_000,
     "market_cap": 85_000_000_000, "revenue": 127_000_000_000,
     "interest_expense": 1_700_000_000, "accounts_payable": 21_000_000_000,
     "accounts_receivable": 12_500_000_000, "cogs": 113_000_000_000,
     "long_term_debt": 68_000_000_000, "stockholders_equity": 44_000_000_000},
    {"ticker": "GM",  "name": "General Motors", "year": 2022,
     "total_assets": 250_000_000_000, "current_assets": 76_000_000_000,
     "current_liabilities": 73_000_000_000, "total_liabilities": 206_000_000_000,
     "retained_earnings": 30_000_000_000, "ebit": 9_400_000_000,
     "market_cap": 55_000_000_000, "revenue": 157_000_000_000,
     "interest_expense": 2_200_000_000, "accounts_payable": 23_000_000_000,
     "accounts_receivable": 13_000_000_000, "cogs": 140_000_000_000,
     "long_term_debt": 66_000_000_000, "stockholders_equity": 44_000_000_000},
    {"ticker": "GM",  "name": "General Motors", "year": 2023,
     "total_assets": 265_000_000_000, "current_assets": 80_000_000_000,
     "current_liabilities": 74_000_000_000, "total_liabilities": 218_000_000_000,
     "retained_earnings": 33_000_000_000, "ebit": 9_800_000_000,
     "market_cap": 45_000_000_000, "revenue": 172_000_000_000,
     "interest_expense": 2_800_000_000, "accounts_payable": 24_500_000_000,
     "accounts_receivable": 14_000_000_000, "cogs": 152_000_000_000,
     "long_term_debt": 70_000_000_000, "stockholders_equity": 47_000_000_000},

    # ---- Stellantis (STLA) ----
    {"ticker": "STLA", "name": "Stellantis", "year": 2021,
     "total_assets": 185_000_000_000, "current_assets": 70_000_000_000,
     "current_liabilities": 55_000_000_000, "total_liabilities": 130_000_000_000,
     "retained_earnings": 25_000_000_000, "ebit": 14_000_000_000,
     "market_cap": 50_000_000_000, "revenue": 152_000_000_000,
     "interest_expense": 1_200_000_000, "accounts_payable": 28_000_000_000,
     "accounts_receivable": 12_000_000_000, "cogs": 120_000_000_000,
     "long_term_debt": 28_000_000_000, "stockholders_equity": 55_000_000_000},
    {"ticker": "STLA", "name": "Stellantis", "year": 2022,
     "total_assets": 190_000_000_000, "current_assets": 72_000_000_000,
     "current_liabilities": 58_000_000_000, "total_liabilities": 134_000_000_000,
     "retained_earnings": 30_000_000_000, "ebit": 16_800_000_000,
     "market_cap": 55_000_000_000, "revenue": 179_000_000_000,
     "interest_expense": 1_400_000_000, "accounts_payable": 30_000_000_000,
     "accounts_receivable": 13_000_000_000, "cogs": 142_000_000_000,
     "long_term_debt": 29_000_000_000, "stockholders_equity": 56_000_000_000},
    {"ticker": "STLA", "name": "Stellantis", "year": 2023,
     "total_assets": 195_000_000_000, "current_assets": 74_000_000_000,
     "current_liabilities": 60_000_000_000, "total_liabilities": 138_000_000_000,
     "retained_earnings": 34_000_000_000, "ebit": 15_500_000_000,
     "market_cap": 62_000_000_000, "revenue": 189_000_000_000,
     "interest_expense": 1_600_000_000, "accounts_payable": 31_000_000_000,
     "accounts_receivable": 13_500_000_000, "cogs": 150_000_000_000,
     "long_term_debt": 30_000_000_000, "stockholders_equity": 57_000_000_000},

    # ---- Aptiv (APTV) ---- Tier-2 electrical systems
    {"ticker": "APTV", "name": "Aptiv", "year": 2019,
     "total_assets": 10_200_000_000, "current_assets": 4_200_000_000,
     "current_liabilities": 3_000_000_000, "total_liabilities": 8_000_000_000,
     "retained_earnings": 1_500_000_000, "ebit": 1_100_000_000,
     "market_cap": 15_000_000_000, "revenue": 14_400_000_000,
     "interest_expense": 250_000_000, "accounts_payable": 2_400_000_000,
     "accounts_receivable": 2_800_000_000, "cogs": 11_500_000_000,
     "long_term_debt": 3_600_000_000, "stockholders_equity": 2_200_000_000},
    {"ticker": "APTV", "name": "Aptiv", "year": 2020,
     "total_assets": 10_600_000_000, "current_assets": 4_500_000_000,
     "current_liabilities": 3_300_000_000, "total_liabilities": 8_400_000_000,
     "retained_earnings": 1_200_000_000, "ebit": 600_000_000,
     "market_cap": 20_000_000_000, "revenue": 13_000_000_000,
     "interest_expense": 260_000_000, "accounts_payable": 2_000_000_000,
     "accounts_receivable": 2_500_000_000, "cogs": 10_500_000_000,
     "long_term_debt": 3_700_000_000, "stockholders_equity": 2_200_000_000},
    {"ticker": "APTV", "name": "Aptiv", "year": 2021,
     "total_assets": 12_000_000_000, "current_assets": 5_000_000_000,
     "current_liabilities": 3_500_000_000, "total_liabilities": 9_000_000_000,
     "retained_earnings": 1_500_000_000, "ebit": 800_000_000,
     "market_cap": 35_000_000_000, "revenue": 15_600_000_000,
     "interest_expense": 270_000_000, "accounts_payable": 2_600_000_000,
     "accounts_receivable": 3_000_000_000, "cogs": 12_700_000_000,
     "long_term_debt": 3_800_000_000, "stockholders_equity": 3_000_000_000},
    {"ticker": "APTV", "name": "Aptiv", "year": 2022,
     "total_assets": 13_500_000_000, "current_assets": 5_500_000_000,
     "current_liabilities": 3_800_000_000, "total_liabilities": 10_000_000_000,
     "retained_earnings": 1_800_000_000, "ebit": 1_000_000_000,
     "market_cap": 25_000_000_000, "revenue": 17_500_000_000,
     "interest_expense": 350_000_000, "accounts_payable": 2_800_000_000,
     "accounts_receivable": 3_200_000_000, "cogs": 14_200_000_000,
     "long_term_debt": 4_000_000_000, "stockholders_equity": 3_500_000_000},
    {"ticker": "APTV", "name": "Aptiv", "year": 2023,
     "total_assets": 14_200_000_000, "current_assets": 5_700_000_000,
     "current_liabilities": 4_000_000_000, "total_liabilities": 10_600_000_000,
     "retained_earnings": 2_000_000_000, "ebit": 1_200_000_000,
     "market_cap": 20_000_000_000, "revenue": 19_800_000_000,
     "interest_expense": 400_000_000, "accounts_payable": 3_000_000_000,
     "accounts_receivable": 3_500_000_000, "cogs": 16_000_000_000,
     "long_term_debt": 4_200_000_000, "stockholders_equity": 3_600_000_000},

    # ---- BorgWarner (BWA) ---- Tier-2, powertrain/EV components
    {"ticker": "BWA",  "name": "BorgWarner", "year": 2019,
     "total_assets": 10_300_000_000, "current_assets": 3_600_000_000,
     "current_liabilities": 2_500_000_000, "total_liabilities": 6_800_000_000,
     "retained_earnings": 3_000_000_000, "ebit": 1_100_000_000,
     "market_cap": 8_000_000_000, "revenue": 9_800_000_000,
     "interest_expense": 190_000_000, "accounts_payable": 1_400_000_000,
     "accounts_receivable": 1_900_000_000, "cogs": 7_800_000_000,
     "long_term_debt": 2_700_000_000, "stockholders_equity": 3_500_000_000},
    {"ticker": "BWA",  "name": "BorgWarner", "year": 2020,
     "total_assets": 14_000_000_000, "current_assets": 4_200_000_000,  # Delphi acquisition
     "current_liabilities": 3_000_000_000, "total_liabilities": 9_500_000_000,
     "retained_earnings": 2_800_000_000, "ebit": 400_000_000,
     "market_cap": 9_500_000_000, "revenue": 8_700_000_000,
     "interest_expense": 320_000_000, "accounts_payable": 1_200_000_000,
     "accounts_receivable": 1_800_000_000, "cogs": 7_000_000_000,
     "long_term_debt": 4_700_000_000, "stockholders_equity": 4_500_000_000},
    {"ticker": "BWA",  "name": "BorgWarner", "year": 2021,
     "total_assets": 14_800_000_000, "current_assets": 4_500_000_000,
     "current_liabilities": 3_200_000_000, "total_liabilities": 9_800_000_000,
     "retained_earnings": 3_100_000_000, "ebit": 900_000_000,
     "market_cap": 11_000_000_000, "revenue": 14_800_000_000,
     "interest_expense": 330_000_000, "accounts_payable": 1_800_000_000,
     "accounts_receivable": 2_300_000_000, "cogs": 12_000_000_000,
     "long_term_debt": 4_500_000_000, "stockholders_equity": 5_000_000_000},
    {"ticker": "BWA",  "name": "BorgWarner", "year": 2022,
     "total_assets": 16_000_000_000, "current_assets": 4_800_000_000,
     "current_liabilities": 3_600_000_000, "total_liabilities": 10_500_000_000,
     "retained_earnings": 3_300_000_000, "ebit": 950_000_000,
     "market_cap": 9_000_000_000, "revenue": 15_800_000_000,
     "interest_expense": 420_000_000, "accounts_payable": 2_000_000_000,
     "accounts_receivable": 2_500_000_000, "cogs": 12_800_000_000,
     "long_term_debt": 4_600_000_000, "stockholders_equity": 5_500_000_000},
    {"ticker": "BWA",  "name": "BorgWarner", "year": 2023,
     "total_assets": 17_000_000_000, "current_assets": 5_000_000_000,
     "current_liabilities": 3_800_000_000, "total_liabilities": 11_200_000_000,
     "retained_earnings": 3_500_000_000, "ebit": 1_100_000_000,
     "market_cap": 7_500_000_000, "revenue": 14_200_000_000,
     "interest_expense": 480_000_000, "accounts_payable": 2_100_000_000,
     "accounts_receivable": 2_600_000_000, "cogs": 11_500_000_000,
     "long_term_debt": 4_800_000_000, "stockholders_equity": 5_800_000_000},

    # ---- Magna International (MGA) ---- Tier-2, body/chassis
    {"ticker": "MGA",  "name": "Magna International", "year": 2019,
     "total_assets": 19_000_000_000, "current_assets": 7_500_000_000,
     "current_liabilities": 5_500_000_000, "total_liabilities": 13_000_000_000,
     "retained_earnings": 5_000_000_000, "ebit": 1_700_000_000,
     "market_cap": 15_000_000_000, "revenue": 39_400_000_000,
     "interest_expense": 130_000_000, "accounts_payable": 4_500_000_000,
     "accounts_receivable": 4_900_000_000, "cogs": 34_500_000_000,
     "long_term_debt": 3_500_000_000, "stockholders_equity": 6_000_000_000},
    {"ticker": "MGA",  "name": "Magna International", "year": 2020,
     "total_assets": 19_500_000_000, "current_assets": 7_200_000_000,
     "current_liabilities": 5_800_000_000, "total_liabilities": 13_500_000_000,
     "retained_earnings": 4_500_000_000, "ebit": 800_000_000,
     "market_cap": 15_500_000_000, "revenue": 32_600_000_000,
     "interest_expense": 140_000_000, "accounts_payable": 3_800_000_000,
     "accounts_receivable": 4_000_000_000, "cogs": 28_500_000_000,
     "long_term_debt": 3_700_000_000, "stockholders_equity": 6_000_000_000},
    {"ticker": "MGA",  "name": "Magna International", "year": 2021,
     "total_assets": 21_000_000_000, "current_assets": 7_800_000_000,
     "current_liabilities": 6_200_000_000, "total_liabilities": 14_500_000_000,
     "retained_earnings": 4_800_000_000, "ebit": 900_000_000,
     "market_cap": 20_000_000_000, "revenue": 36_200_000_000,
     "interest_expense": 155_000_000, "accounts_payable": 4_200_000_000,
     "accounts_receivable": 4_600_000_000, "cogs": 32_000_000_000,
     "long_term_debt": 3_900_000_000, "stockholders_equity": 6_500_000_000},
    {"ticker": "MGA",  "name": "Magna International", "year": 2022,
     "total_assets": 22_000_000_000, "current_assets": 8_000_000_000,
     "current_liabilities": 6_500_000_000, "total_liabilities": 15_000_000_000,
     "retained_earnings": 5_000_000_000, "ebit": 1_100_000_000,
     "market_cap": 15_000_000_000, "revenue": 37_800_000_000,
     "interest_expense": 200_000_000, "accounts_payable": 4_400_000_000,
     "accounts_receivable": 4_800_000_000, "cogs": 33_500_000_000,
     "long_term_debt": 4_000_000_000, "stockholders_equity": 7_000_000_000},
    {"ticker": "MGA",  "name": "Magna International", "year": 2023,
     "total_assets": 23_000_000_000, "current_assets": 8_200_000_000,
     "current_liabilities": 6_600_000_000, "total_liabilities": 15_500_000_000,
     "retained_earnings": 5_200_000_000, "ebit": 1_300_000_000,
     "market_cap": 13_000_000_000, "revenue": 42_800_000_000,
     "interest_expense": 230_000_000, "accounts_payable": 4_700_000_000,
     "accounts_receivable": 5_100_000_000, "cogs": 37_800_000_000,
     "long_term_debt": 4_200_000_000, "stockholders_equity": 7_500_000_000},

    # ---- Adient (ADNT) ---- Tier-3, seating systems (typically high leverage)
    {"ticker": "ADNT", "name": "Adient", "year": 2019,
     "total_assets": 8_000_000_000, "current_assets": 2_700_000_000,
     "current_liabilities": 2_500_000_000, "total_liabilities": 6_700_000_000,
     "retained_earnings": -1_500_000_000, "ebit": 300_000_000,
     "market_cap": 2_500_000_000, "revenue": 17_400_000_000,
     "interest_expense": 280_000_000, "accounts_payable": 2_200_000_000,
     "accounts_receivable": 1_500_000_000, "cogs": 15_800_000_000,
     "long_term_debt": 3_500_000_000, "stockholders_equity": 1_300_000_000},
    {"ticker": "ADNT", "name": "Adient", "year": 2020,
     "total_assets": 7_500_000_000, "current_assets": 2_500_000_000,
     "current_liabilities": 2_400_000_000, "total_liabilities": 6_500_000_000,
     "retained_earnings": -2_000_000_000, "ebit": -200_000_000,
     "market_cap": 1_800_000_000, "revenue": 14_600_000_000,
     "interest_expense": 290_000_000, "accounts_payable": 1_800_000_000,
     "accounts_receivable": 1_200_000_000, "cogs": 13_400_000_000,
     "long_term_debt": 3_400_000_000, "stockholders_equity": 1_000_000_000},
    {"ticker": "ADNT", "name": "Adient", "year": 2021,
     "total_assets": 7_800_000_000, "current_assets": 2_600_000_000,
     "current_liabilities": 2_500_000_000, "total_liabilities": 6_600_000_000,
     "retained_earnings": -1_800_000_000, "ebit": 400_000_000,
     "market_cap": 3_500_000_000, "revenue": 15_600_000_000,
     "interest_expense": 280_000_000, "accounts_payable": 2_000_000_000,
     "accounts_receivable": 1_400_000_000, "cogs": 14_200_000_000,
     "long_term_debt": 3_300_000_000, "stockholders_equity": 1_200_000_000},
    {"ticker": "ADNT", "name": "Adient", "year": 2022,
     "total_assets": 7_600_000_000, "current_assets": 2_500_000_000,
     "current_liabilities": 2_600_000_000, "total_liabilities": 6_400_000_000,
     "retained_earnings": -1_600_000_000, "ebit": 350_000_000,
     "market_cap": 2_800_000_000, "revenue": 15_400_000_000,
     "interest_expense": 300_000_000, "accounts_payable": 2_100_000_000,
     "accounts_receivable": 1_300_000_000, "cogs": 14_000_000_000,
     "long_term_debt": 3_200_000_000, "stockholders_equity": 1_200_000_000},
    {"ticker": "ADNT", "name": "Adient", "year": 2023,
     "total_assets": 7_800_000_000, "current_assets": 2_600_000_000,
     "current_liabilities": 2_700_000_000, "total_liabilities": 6_500_000_000,
     "retained_earnings": -1_400_000_000, "ebit": 450_000_000,
     "market_cap": 2_200_000_000, "revenue": 16_300_000_000,
     "interest_expense": 280_000_000, "accounts_payable": 2_200_000_000,
     "accounts_receivable": 1_400_000_000, "cogs": 14_800_000_000,
     "long_term_debt": 3_100_000_000, "stockholders_equity": 1_300_000_000},

    # ---- Lear Corporation (LEA) ---- Tier-3, seating + electrical
    {"ticker": "LEA",  "name": "Lear Corporation", "year": 2019,
     "total_assets": 10_000_000_000, "current_assets": 4_000_000_000,
     "current_liabilities": 3_200_000_000, "total_liabilities": 6_800_000_000,
     "retained_earnings": 3_000_000_000, "ebit": 1_000_000_000,
     "market_cap": 8_500_000_000, "revenue": 19_800_000_000,
     "interest_expense": 130_000_000, "accounts_payable": 2_500_000_000,
     "accounts_receivable": 2_800_000_000, "cogs": 17_500_000_000,
     "long_term_debt": 2_100_000_000, "stockholders_equity": 3_200_000_000},
    {"ticker": "LEA",  "name": "Lear Corporation", "year": 2020,
     "total_assets": 9_500_000_000, "current_assets": 3_700_000_000,
     "current_liabilities": 3_000_000_000, "total_liabilities": 6_500_000_000,
     "retained_earnings": 2_700_000_000, "ebit": 300_000_000,
     "market_cap": 8_000_000_000, "revenue": 17_000_000_000,
     "interest_expense": 140_000_000, "accounts_payable": 2_000_000_000,
     "accounts_receivable": 2_400_000_000, "cogs": 15_000_000_000,
     "long_term_debt": 2_300_000_000, "stockholders_equity": 3_000_000_000},
    {"ticker": "LEA",  "name": "Lear Corporation", "year": 2021,
     "total_assets": 10_500_000_000, "current_assets": 4_100_000_000,
     "current_liabilities": 3_300_000_000, "total_liabilities": 7_000_000_000,
     "retained_earnings": 3_100_000_000, "ebit": 700_000_000,
     "market_cap": 10_000_000_000, "revenue": 19_200_000_000,
     "interest_expense": 145_000_000, "accounts_payable": 2_400_000_000,
     "accounts_receivable": 2_600_000_000, "cogs": 17_000_000_000,
     "long_term_debt": 2_200_000_000, "stockholders_equity": 3_500_000_000},
    {"ticker": "LEA",  "name": "Lear Corporation", "year": 2022,
     "total_assets": 10_800_000_000, "current_assets": 4_200_000_000,
     "current_liabilities": 3_500_000_000, "total_liabilities": 7_200_000_000,
     "retained_earnings": 3_200_000_000, "ebit": 800_000_000,
     "market_cap": 7_500_000_000, "revenue": 20_900_000_000,
     "interest_expense": 180_000_000, "accounts_payable": 2_600_000_000,
     "accounts_receivable": 2_700_000_000, "cogs": 18_500_000_000,
     "long_term_debt": 2_300_000_000, "stockholders_equity": 3_600_000_000},
    {"ticker": "LEA",  "name": "Lear Corporation", "year": 2023,
     "total_assets": 11_200_000_000, "current_assets": 4_400_000_000,
     "current_liabilities": 3_600_000_000, "total_liabilities": 7_500_000_000,
     "retained_earnings": 3_400_000_000, "ebit": 900_000_000,
     "market_cap": 6_800_000_000, "revenue": 22_900_000_000,
     "interest_expense": 200_000_000, "accounts_payable": 2_800_000_000,
     "accounts_receivable": 2_900_000_000, "cogs": 20_200_000_000,
     "long_term_debt": 2_400_000_000, "stockholders_equity": 3_700_000_000},

    # ---- Dana Incorporated (DAN) ---- Tier-3, driveline systems
    {"ticker": "DAN",  "name": "Dana Incorporated", "year": 2019,
     "total_assets": 6_800_000_000, "current_assets": 2_200_000_000,
     "current_liabilities": 1_900_000_000, "total_liabilities": 5_200_000_000,
     "retained_earnings": 500_000_000, "ebit": 450_000_000,
     "market_cap": 2_200_000_000, "revenue": 8_600_000_000,
     "interest_expense": 185_000_000, "accounts_payable": 1_100_000_000,
     "accounts_receivable": 1_400_000_000, "cogs": 7_400_000_000,
     "long_term_debt": 2_500_000_000, "stockholders_equity": 1_600_000_000},
    {"ticker": "DAN",  "name": "Dana Incorporated", "year": 2020,
     "total_assets": 6_600_000_000, "current_assets": 2_000_000_000,
     "current_liabilities": 1_800_000_000, "total_liabilities": 5_000_000_000,
     "retained_earnings": 300_000_000, "ebit": 150_000_000,
     "market_cap": 1_800_000_000, "revenue": 7_100_000_000,
     "interest_expense": 200_000_000, "accounts_payable": 900_000_000,
     "accounts_receivable": 1_100_000_000, "cogs": 6_100_000_000,
     "long_term_debt": 2_400_000_000, "stockholders_equity": 1_600_000_000},
    {"ticker": "DAN",  "name": "Dana Incorporated", "year": 2021,
     "total_assets": 7_000_000_000, "current_assets": 2_300_000_000,
     "current_liabilities": 2_000_000_000, "total_liabilities": 5_400_000_000,
     "retained_earnings": 400_000_000, "ebit": 350_000_000,
     "market_cap": 2_500_000_000, "revenue": 8_900_000_000,
     "interest_expense": 195_000_000, "accounts_payable": 1_100_000_000,
     "accounts_receivable": 1_300_000_000, "cogs": 7_700_000_000,
     "long_term_debt": 2_500_000_000, "stockholders_equity": 1_600_000_000},
    {"ticker": "DAN",  "name": "Dana Incorporated", "year": 2022,
     "total_assets": 7_300_000_000, "current_assets": 2_400_000_000,
     "current_liabilities": 2_100_000_000, "total_liabilities": 5_600_000_000,
     "retained_earnings": 450_000_000, "ebit": 300_000_000,
     "market_cap": 1_500_000_000, "revenue": 10_600_000_000,
     "interest_expense": 230_000_000, "accounts_payable": 1_300_000_000,
     "accounts_receivable": 1_500_000_000, "cogs": 9_400_000_000,
     "long_term_debt": 2_600_000_000, "stockholders_equity": 1_700_000_000},
    {"ticker": "DAN",  "name": "Dana Incorporated", "year": 2023,
     "total_assets": 7_500_000_000, "current_assets": 2_500_000_000,
     "current_liabilities": 2_200_000_000, "total_liabilities": 5_700_000_000,
     "retained_earnings": 500_000_000, "ebit": 350_000_000,
     "market_cap": 1_200_000_000, "revenue": 10_600_000_000,
     "interest_expense": 260_000_000, "accounts_payable": 1_400_000_000,
     "accounts_receivable": 1_500_000_000, "cogs": 9_400_000_000,
     "long_term_debt": 2_700_000_000, "stockholders_equity": 1_800_000_000},

    # ---- Modine Manufacturing (MOD) ---- Tier-3, thermal management
    {"ticker": "MOD",  "name": "Modine Manufacturing", "year": 2019,
     "total_assets": 1_600_000_000, "current_assets": 650_000_000,
     "current_liabilities": 480_000_000, "total_liabilities": 1_100_000_000,
     "retained_earnings": 150_000_000, "ebit": 80_000_000,
     "market_cap": 550_000_000, "revenue": 2_100_000_000,
     "interest_expense": 35_000_000, "accounts_payable": 280_000_000,
     "accounts_receivable": 380_000_000, "cogs": 1_800_000_000,
     "long_term_debt": 450_000_000, "stockholders_equity": 500_000_000},
    {"ticker": "MOD",  "name": "Modine Manufacturing", "year": 2020,
     "total_assets": 1_550_000_000, "current_assets": 600_000_000,
     "current_liabilities": 460_000_000, "total_liabilities": 1_050_000_000,
     "retained_earnings": 100_000_000, "ebit": 40_000_000,
     "market_cap": 400_000_000, "revenue": 1_800_000_000,
     "interest_expense": 38_000_000, "accounts_payable": 240_000_000,
     "accounts_receivable": 320_000_000, "cogs": 1_550_000_000,
     "long_term_debt": 430_000_000, "stockholders_equity": 500_000_000},
    {"ticker": "MOD",  "name": "Modine Manufacturing", "year": 2021,
     "total_assets": 1_600_000_000, "current_assets": 630_000_000,
     "current_liabilities": 490_000_000, "total_liabilities": 1_100_000_000,
     "retained_earnings": 120_000_000, "ebit": 60_000_000,
     "market_cap": 600_000_000, "revenue": 2_000_000_000,
     "interest_expense": 36_000_000, "accounts_payable": 260_000_000,
     "accounts_receivable": 350_000_000, "cogs": 1_720_000_000,
     "long_term_debt": 420_000_000, "stockholders_equity": 500_000_000},
    {"ticker": "MOD",  "name": "Modine Manufacturing", "year": 2022,
     "total_assets": 1_750_000_000, "current_assets": 680_000_000,
     "current_liabilities": 510_000_000, "total_liabilities": 1_150_000_000,
     "retained_earnings": 160_000_000, "ebit": 100_000_000,
     "market_cap": 850_000_000, "revenue": 2_300_000_000,
     "interest_expense": 40_000_000, "accounts_payable": 290_000_000,
     "accounts_receivable": 380_000_000, "cogs": 1_980_000_000,
     "long_term_debt": 390_000_000, "stockholders_equity": 600_000_000},
    {"ticker": "MOD",  "name": "Modine Manufacturing", "year": 2023,
     "total_assets": 1_900_000_000, "current_assets": 720_000_000,
     "current_liabilities": 530_000_000, "total_liabilities": 1_200_000_000,
     "retained_earnings": 200_000_000, "ebit": 140_000_000,
     "market_cap": 1_800_000_000, "revenue": 2_400_000_000,
     "interest_expense": 42_000_000, "accounts_payable": 310_000_000,
     "accounts_receivable": 400_000_000, "cogs": 2_050_000_000,
     "long_term_debt": 370_000_000, "stockholders_equity": 700_000_000},
]


def seed_all(config: dict) -> None:
    """
    Generate synthetic data files and run the ratio calculator on them.

    Creates:
      - data/financials/{ticker}.parquet  (one per firm)
      - data/financials/all_firms.parquet (combined)
      - data/supply_chain/edges.csv
      - risk_framework/scores.csv
      - data_quality_report.md

    Parameters
    ----------
    config : dict
        Loaded config.yaml.
    """
    from data_engineering.ratio_calculator import compute_ratios
    from data_engineering.supply_chain_builder import build_edges
    from data_engineering.data_quality_report import generate_report
    from data_engineering.macro_fetcher import fetch_macro

    fin_dir = ROOT / config["data"]["financials_dir"]
    fin_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(SEED_DATA)
    print(f"Seeding {df['ticker'].nunique()} firms, {len(df)} firm-years...")

    # Save per-ticker parquets
    for ticker, group in df.groupby("ticker"):
        path = fin_dir / f"{ticker}.parquet"
        group.to_parquet(path, index=False)
        print(f"  Saved {path}")

    # Save combined
    all_path = fin_dir / "all_firms.parquet"
    df.to_parquet(all_path, index=False)
    print(f"  Saved {all_path}")

    # Compute ratios
    df_scored = compute_ratios(df, config)
    scores_path = ROOT / "risk_framework" / "scores.csv"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    df_scored.to_csv(scores_path, index=False)
    print(f"  Saved {scores_path}")

    # Build supply chain edges
    edges_df = build_edges(config)
    print(f"  Saved {len(edges_df)} edges to data/supply_chain/edges.csv")

    # Generate quality report
    generate_report(df_scored, config, str(ROOT / "data_quality_report.md"))
    print(f"  Saved data_quality_report.md")

    # Fetch/generate macro data
    macro_df = fetch_macro(config)
    print(f"  Saved {len(macro_df)} macro observations to data/macro/macro_series.csv")

    print("\n✅ Demo data seeded. Run: streamlit run dashboard/app.py")


if __name__ == "__main__":
    seed_all(config)
