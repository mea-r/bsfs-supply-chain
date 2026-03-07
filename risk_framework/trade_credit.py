"""
trade_credit.py — Trade credit exposure quantification.

Computes trade credit at risk for a financial institution exposed to
supply chain firms through trade credit or working-capital financing.

Economic framework:
  Trade credit exposure = Accounts Receivable from stressed counterparties.
  Expected Loss = Exposure × Probability of Default (PD) × Loss Given Default (LGD).

  PD is derived from the Altman Z-Score using the empirical mapping:
    - Safe zone (Z > 2.99):    PD ≈ 0.2%  (baseline, almost zero)
    - Grey zone (1.81-2.99):   PD ≈ 5-15% (elevated, uncertain)
    - Distress zone (Z ≤ 1.81): PD ≈ 30-70% (very high)

  LGD for unsecured trade credit is typically 60% (Moody's 2023 averages
  for senior unsecured claims in manufacturing).
"""

import pandas as pd
import numpy as np


# PD mapping from Z-Score (piecewise linear)
def estimate_pd(z_score: float) -> float:
    """
    Estimate annualized probability of default from Altman Z-Score.

    Uses a piecewise linear mapping calibrated to Altman's original study
    and updated with modern default data (Moody's 2023).

    Parameters
    ----------
    z_score : float
        Altman Z-Score.

    Returns
    -------
    float
        Estimated PD in [0, 1].
    """
    if pd.isna(z_score):
        return 0.10  # conservative estimate for unknown
    if z_score > 2.99:
        return 0.002
    if z_score > 1.81:
        # Linear interpolation: 2.99 → 2%, 1.81 → 15%
        return 0.02 + (2.99 - z_score) / (2.99 - 1.81) * (0.15 - 0.02)
    if z_score > 0:
        # Linear interpolation: 1.81 → 15%, 0 → 50%
        return 0.15 + (1.81 - z_score) / 1.81 * (0.50 - 0.15)
    # Below 0: very high PD
    return min(0.70, 0.50 + abs(z_score) * 0.05)


# Default LGD for unsecured trade credit (Moody's 2023)
DEFAULT_LGD = 0.60


def compute_trade_credit_exposure(
    node_states: dict,
    edges_df: pd.DataFrame,
    lgd: float = DEFAULT_LGD,
) -> pd.DataFrame:
    """
    Compute trade credit exposure for each firm in the supply chain.

    For each firm, estimates:
      - Accounts Receivable at risk (from stressed counterparties)
      - Probability of default (from Z-Score)
      - Expected loss = AR × PD × LGD
      - Trade credit extended (DPO-implied payables)

    Parameters
    ----------
    node_states : dict
        Current node states from PropagationEngine.
    edges_df : pd.DataFrame
        Supply chain edges.
    lgd : float
        Loss given default (default: 0.60).

    Returns
    -------
    pd.DataFrame
        One row per firm with exposure metrics.
    """
    rows = []
    for ticker, state in node_states.items():
        z = state.get("z_score", float("nan"))
        ar = state.get("accounts_receivable", float("nan"))
        ap = state.get("accounts_payable", float("nan"))
        rev = state.get("revenue", float("nan"))
        zone = state.get("credit_zone", "unknown")
        stress = state.get("stress_score", 0.0)
        pd_est = estimate_pd(z)

        # AR at risk = total AR × PD (simplified: assumes all AR is equally at risk)
        ar_at_risk = ar * pd_est if not pd.isna(ar) else float("nan")

        # Expected loss
        expected_loss = ar_at_risk * lgd if not pd.isna(ar_at_risk) else float("nan")

        # Count downstream buyers and upstream suppliers
        n_buyers = len(edges_df[edges_df["source"] == ticker])
        n_suppliers = len(edges_df[edges_df["target"] == ticker])

        rows.append({
            "ticker": ticker,
            "name": state.get("name", ticker),
            "credit_zone": zone,
            "z_score": round(z, 4) if not pd.isna(z) else float("nan"),
            "stress_score": round(stress, 4),
            "pd_estimate": round(pd_est, 4),
            "accounts_receivable": ar,
            "ar_at_risk": round(ar_at_risk, 0) if not pd.isna(ar_at_risk) else float("nan"),
            "expected_loss": round(expected_loss, 0) if not pd.isna(expected_loss) else float("nan"),
            "lgd": lgd,
            "accounts_payable": ap,
            "n_buyers": n_buyers,
            "n_suppliers": n_suppliers,
        })

    df = pd.DataFrame(rows)
    return df.sort_values("expected_loss", ascending=False, na_position="last").reset_index(drop=True)


def portfolio_summary(exposure_df: pd.DataFrame) -> dict:
    """
    Compute aggregate portfolio-level trade credit exposure.

    Parameters
    ----------
    exposure_df : pd.DataFrame
        Output of compute_trade_credit_exposure().

    Returns
    -------
    dict
        Portfolio-level metrics.
    """
    total_ar = exposure_df["accounts_receivable"].sum()
    total_ar_at_risk = exposure_df["ar_at_risk"].sum()
    total_expected_loss = exposure_df["expected_loss"].sum()
    n_distressed = (exposure_df["credit_zone"] == "distress").sum()
    n_grey = (exposure_df["credit_zone"] == "grey").sum()
    avg_pd = exposure_df["pd_estimate"].mean()

    return {
        "total_accounts_receivable": total_ar,
        "total_ar_at_risk": total_ar_at_risk,
        "total_expected_loss": total_expected_loss,
        "pct_ar_at_risk": round(total_ar_at_risk / total_ar * 100, 2) if total_ar > 0 else 0,
        "avg_pd": round(avg_pd, 4),
        "n_distressed_firms": int(n_distressed),
        "n_grey_zone_firms": int(n_grey),
        "n_total_firms": len(exposure_df),
    }
