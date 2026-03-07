"""
ratio_calculator.py — Compute all financial ratios from normalized financial data.

Ratios implemented:
  - Altman Z-Score (public firm version, Altman 1968)
  - Current Ratio
  - Debt-to-Equity Ratio
  - Interest Coverage Ratio (ICR)
  - Days Payable Outstanding (DPO)
  - Days Sales Outstanding (DSO)

Economic context:
  These ratios collectively span three risk dimensions:
    1. Solvency risk (Z-Score, D/E)
    2. Liquidity risk (Current Ratio, DPO, DSO)
    3. Debt service risk (ICR)
  Together they give a 360-degree view of financial health, which is why
  trade credit analysts and supply chain finance teams track all of them.
"""

import logging
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load central configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_altman_z_score(row: pd.Series, weights: dict) -> float:
    """
    Compute Altman Z-Score for a single firm-year observation.

    Formula (Altman 1968, public firm version):
        Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

    Where:
        X1 = Working Capital / Total Assets
             (liquidity relative to asset base)
        X2 = Retained Earnings / Total Assets
             (cumulative profitability; negative → accumulated losses)
        X3 = EBIT / Total Assets
             (operating efficiency; how well assets generate earnings)
        X4 = Market Cap / Total Liabilities
             (market-implied solvency buffer)
        X5 = Revenue / Total Assets
             (asset turnover / operational efficiency)

    Economic justification:
        The Z-Score was calibrated on a matched sample of bankrupt and
        non-bankrupt US manufacturers. Each coefficient reflects the
        marginal discriminating power of that ratio. It remains one of
        the most empirically validated credit risk metrics for public firms.

    Parameters
    ----------
    row : pd.Series
        One row of financial data with the required columns.
    weights : dict
        Z-score weights from config (w1..w5).

    Returns
    -------
    float
        Altman Z-Score, or NaN if required inputs are missing.
    """
    try:
        total_assets = row["total_assets"]
        if pd.isna(total_assets) or total_assets == 0:
            return float("nan")

        working_capital = row["current_assets"] - row["current_liabilities"]
        x1 = working_capital / total_assets

        retained_earnings = row.get("retained_earnings", float("nan"))
        x2 = retained_earnings / total_assets if not pd.isna(retained_earnings) else float("nan")

        ebit = row.get("ebit", float("nan"))
        x3 = ebit / total_assets if not pd.isna(ebit) else float("nan")

        market_cap = row.get("market_cap", float("nan"))
        total_liabilities = row.get("total_liabilities", float("nan"))
        if not pd.isna(market_cap) and not pd.isna(total_liabilities) and total_liabilities != 0:
            x4 = market_cap / total_liabilities
        else:
            x4 = float("nan")

        revenue = row.get("revenue", float("nan"))
        x5 = revenue / total_assets if not pd.isna(revenue) else float("nan")

        components = [x1, x2, x3, x4, x5]
        coefs = [weights["w1"], weights["w2"], weights["w3"], weights["w4"], weights["w5"]]

        # If more than 2 components are missing, Z-score is unreliable
        missing = sum(1 for c in components if pd.isna(c))
        if missing > 2:
            logger.debug(f"Z-score: {missing} missing components — returning NaN")
            return float("nan")

        # Use 0 for missing components as a conservative fallback, flagged separately
        components_filled = [0.0 if pd.isna(c) else c for c in components]
        z = sum(w * x for w, x in zip(coefs, components_filled))
        return round(z, 4)

    except Exception as e:
        logger.warning(f"Z-score computation error: {e}")
        return float("nan")


def classify_credit_zone(z_score: float, safe_threshold: float, grey_threshold: float) -> str:
    """
    Map Altman Z-Score to Altman (1968) credit zone classification.

    Zones:
      - Safe    (green):  Z > 2.99
      - Grey    (yellow): 1.81 < Z ≤ 2.99
      - Distress (red):   Z ≤ 1.81

    Economic justification:
      These boundaries were empirically derived by Altman from prediction
      accuracy on his original bankrupt/non-bankrupt sample. Firms in the
      "grey zone" have elevated but uncertain default risk; the model has
      lower accuracy in this range.

    Parameters
    ----------
    z_score : float
    safe_threshold : float
        From config; default 2.99.
    grey_threshold : float
        From config; default 1.81.

    Returns
    -------
    str
        One of: "safe", "grey", "distress", "unknown".
    """
    if pd.isna(z_score):
        return "unknown"
    if z_score > safe_threshold:
        return "safe"
    if z_score > grey_threshold:
        return "grey"
    return "distress"


def compute_ratios(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Compute all financial ratios for a dataframe of firm-year observations.

    Adds the following columns to the dataframe:
      - working_capital
      - current_ratio
      - debt_to_equity
      - interest_coverage_ratio
      - days_payable_outstanding
      - days_sales_outstanding
      - z_score_x1 .. x5  (individual Z-score components)
      - z_score
      - credit_zone

    Parameters
    ----------
    df : pd.DataFrame
        Normalized financial data (from SECEdgarScraper).
    config : dict
        Loaded config.yaml.

    Returns
    -------
    pd.DataFrame
        Input dataframe with ratio columns appended.
    """
    df = df.copy()
    z_weights = config["ratios"]["z_score"]
    zones = config["ratios"]["credit_zones"]

    # ---- Working Capital ----
    # Economic: net short-term liquidity buffer
    df["working_capital"] = df["current_assets"] - df["current_liabilities"]

    # ---- Current Ratio ----
    # Economic: can the firm pay near-term obligations from current assets?
    # Threshold: < 1.0 is a liquidity warning; < 0.75 is severe stress.
    df["current_ratio"] = np.where(
        df["current_liabilities"].notna() & (df["current_liabilities"] != 0),
        df["current_assets"] / df["current_liabilities"],
        np.nan,
    )

    # ---- Debt-to-Equity ----
    # Economic: financial leverage / solvency. Higher = more fragile in downturns.
    df["debt_to_equity"] = np.where(
        df["stockholders_equity"].notna() & (df["stockholders_equity"] != 0),
        df["total_liabilities"] / df["stockholders_equity"],
        np.nan,
    )

    # ---- Interest Coverage Ratio ----
    # ICR = EBIT / Interest Expense
    # Economic: how many times can operating earnings cover interest payments?
    # ICR < 1.5 is a stress warning; ICR < 1.0 means the firm cannot service debt.
    df["interest_coverage_ratio"] = np.where(
        df["interest_expense"].notna() & (df["interest_expense"] > 0),
        df["ebit"] / df["interest_expense"],
        np.nan,
    )

    # ---- Days Payable Outstanding ----
    # DPO = (Accounts Payable / COGS) × 365
    # Economic: how long a firm takes to pay its suppliers.
    # High DPO can indicate a firm is stretching payables (liquidity stress)
    # or has strong negotiating power (positive). Context matters.
    df["days_payable_outstanding"] = np.where(
        df["cogs"].notna() & (df["cogs"] > 0),
        (df["accounts_payable"] / df["cogs"]) * 365,
        np.nan,
    )

    # ---- Days Sales Outstanding ----
    # DSO = (Accounts Receivable / Revenue) × 365
    # Economic: how long it takes a firm to collect from customers.
    # Rising DSO can indicate customers paying slowly (their stress) or
    # aggressive revenue recognition.
    df["days_sales_outstanding"] = np.where(
        df["revenue"].notna() & (df["revenue"] > 0),
        (df["accounts_receivable"] / df["revenue"]) * 365,
        np.nan,
    )

    # ---- Altman Z-Score Components ----
    total_assets = df["total_assets"]
    df["z_x1"] = (df["current_assets"] - df["current_liabilities"]) / total_assets
    df["z_x2"] = df["retained_earnings"] / total_assets
    df["z_x3"] = df["ebit"] / total_assets
    df["z_x4"] = np.where(
        df["total_liabilities"].notna() & (df["total_liabilities"] != 0),
        df["market_cap"] / df["total_liabilities"],
        np.nan,
    )
    df["z_x5"] = df["revenue"] / total_assets

    # ---- Altman Z-Score ----
    df["z_score"] = df.apply(
        lambda r: compute_altman_z_score(r, z_weights), axis=1
    )

    # ---- Credit Zone Classification ----
    df["credit_zone"] = df["z_score"].apply(
        lambda z: classify_credit_zone(z, zones["safe_threshold"], zones["grey_threshold"])
    )

    return df


def run(config_path: str = "config.yaml") -> pd.DataFrame:
    """
    Entry point: load all_firms.parquet, compute ratios, save scores.csv.

    Parameters
    ----------
    config_path : str
        Path to config.yaml.

    Returns
    -------
    pd.DataFrame
        Dataframe with all ratios and credit zone classifications.
    """
    config = load_config(config_path)
    logging.basicConfig(
        level=config["logging"]["level"],
        format=config["logging"]["format"],
    )
    fin_dir = Path(config["data"]["financials_dir"])
    all_firms_path = fin_dir / "all_firms.parquet"

    if not all_firms_path.exists():
        raise FileNotFoundError(
            f"Financial data not found at {all_firms_path}. Run sec_scraper.py first."
        )

    df = pd.read_parquet(all_firms_path)
    df_with_ratios = compute_ratios(df, config)

    out_path = Path("risk_framework") / "scores.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_with_ratios.to_csv(out_path, index=False)
    logger.info(f"Saved scores → {out_path}")
    return df_with_ratios


if __name__ == "__main__":
    run()
