"""
scorer.py — Orchestrates the full risk scoring pipeline.

This module:
  1. Loads individual firm parquet files from data/financials/
  2. Applies ratio_calculator.compute_ratios()
  3. Writes risk_framework/scores.csv

It also provides a ScoreStore class used by the dashboard and propagation
engine to query firm-year risk states without re-computing.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path

from utils.config import load_config

logger = logging.getLogger(__name__)


class ScoreStore:
    """
    In-memory store of pre-computed risk scores for all firms.

    Used by PropagationEngine and the dashboard to quickly look up
    a firm's Z-score, credit zone, and financial ratios for a given year.

    Parameters
    ----------
    scores_csv : str or Path
        Path to risk_framework/scores.csv.
    config : dict
        Loaded config.yaml.
    """

    def __init__(self, scores_csv: str = "risk_framework/scores.csv",
                 config: dict = None):
        self.config = config or load_config()
        self._df = pd.read_csv(scores_csv)
        self._df["year"] = self._df["year"].astype(int)
        # Index for fast lookup
        self._index = self._df.set_index(["ticker", "year"])

    def get_firm_year(self, ticker: str, year: int) -> pd.Series:
        """
        Retrieve all scores for a single firm-year.

        Parameters
        ----------
        ticker : str
        year : int

        Returns
        -------
        pd.Series
            All scored fields for that firm-year, or empty Series if not found.
        """
        try:
            return self._index.loc[(ticker, year)]
        except KeyError:
            logger.warning(f"No score for ({ticker}, {year})")
            return pd.Series(dtype=float)

    def get_latest_year(self, ticker: str) -> pd.Series:
        """Get the most recent year's scores for a firm."""
        firm_data = self._df[self._df["ticker"] == ticker]
        if firm_data.empty:
            return pd.Series(dtype=float)
        latest = firm_data.loc[firm_data["year"].idxmax()]
        return latest

    def get_all_firms(self, year: int = None) -> pd.DataFrame:
        """
        Get scores for all firms, optionally filtered to a single year.

        Parameters
        ----------
        year : int, optional
            If provided, returns only that year's data.

        Returns
        -------
        pd.DataFrame
        """
        if year is not None:
            return self._df[self._df["year"] == year].copy()
        return self._df.copy()

    def get_z_score(self, ticker: str, year: int) -> float:
        """Return Z-score for a specific firm-year."""
        row = self.get_firm_year(ticker, year)
        return float(row.get("z_score", float("nan"))) if not row.empty else float("nan")

    def get_credit_zone(self, ticker: str, year: int) -> str:
        """Return credit zone for a specific firm-year."""
        row = self.get_firm_year(ticker, year)
        return str(row.get("credit_zone", "unknown")) if not row.empty else "unknown"

    def get_tickers(self) -> list:
        """Return list of all tickers in the store."""
        return self._df["ticker"].unique().tolist()

    def get_years(self) -> list:
        """Return sorted list of available years."""
        return sorted(self._df["year"].unique().tolist())

    def summary_stats(self, year: int) -> dict:
        """
        Compute summary statistics for a given year.

        Returns
        -------
        dict
            Counts and percentages by credit zone.
        """
        df_year = self.get_all_firms(year)
        total = len(df_year)
        if total == 0:
            return {}
        zone_counts = df_year["credit_zone"].value_counts().to_dict()
        return {
            "total": total,
            "safe": zone_counts.get("safe", 0),
            "grey": zone_counts.get("grey", 0),
            "distress": zone_counts.get("distress", 0),
            "unknown": zone_counts.get("unknown", 0),
            "pct_safe": round(zone_counts.get("safe", 0) / total * 100, 1),
            "pct_grey": round(zone_counts.get("grey", 0) / total * 100, 1),
            "pct_distress": round(zone_counts.get("distress", 0) / total * 100, 1),
        }


def run(config_path: str = "config.yaml") -> "ScoreStore":
    """
    Entry point: run the full scoring pipeline.

    Loads financials, computes ratios, saves scores.csv, returns ScoreStore.

    Parameters
    ----------
    config_path : str

    Returns
    -------
    ScoreStore
    """
    # Lazy import to avoid circular dependency with data_engineering
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data_engineering.ratio_calculator import run as compute_ratios_run

    config = load_config(config_path)
    logging.basicConfig(
        level=config["logging"]["level"],
        format=config["logging"]["format"],
    )
    compute_ratios_run(config_path)
    return ScoreStore(config=config)


if __name__ == "__main__":
    run()
