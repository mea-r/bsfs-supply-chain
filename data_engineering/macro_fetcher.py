"""
macro_fetcher.py — Fetch macroeconomic time series from FRED (Federal Reserve).

Fetches the FRED series defined in config.yaml and saves them to data/macro/macro_series.csv.

Sources:
  - Federal Reserve Bank of St. Louis FRED API (https://fred.stlouisfed.org/)
  - Series: Fed Funds Rate, CPI, Industrial Production, HY Credit Spread

Requires:
  - FRED_API_KEY environment variable (free at https://fred.stlouisfed.org/docs/api/api_key.html)
  - OR: falls back to bundled CSV if API is unavailable
"""

import logging
import os
from pathlib import Path

import pandas as pd
from utils.config import load_config

logger = logging.getLogger(__name__)


def _fetch_fred_series(
    series_id: str, api_key: str, start: str = "2018-01-01", end: str = "2024-12-31"
) -> pd.DataFrame:
    """
    Fetch a single FRED series via the FRED API.

    Parameters
    ----------
    series_id : str
        FRED series ID (e.g., "FEDFUNDS").
    api_key : str
        FRED API key.
    start : str
        Start date (YYYY-MM-DD).
    end : str
        End date (YYYY-MM-DD).

    Returns
    -------
    pd.DataFrame
        Columns: date, value, series_id
    """
    import requests

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    records = []
    for obs in data.get("observations", []):
        val = obs.get("value", ".")
        if val == ".":
            continue
        records.append(
            {
                "date": obs["date"],
                "value": float(val),
                "series_id": series_id,
            }
        )
    return pd.DataFrame(records)


def _generate_bundled_macro_data(fred_series: dict) -> pd.DataFrame:
    """
    Generate approximate macro data as a fallback when FRED API is unavailable.

    Values are approximate monthly averages calibrated to real FRED data
    for 2019-2023 to support offline demo usage.
    """
    import numpy as np

    dates = pd.date_range("2019-01-01", "2023-12-01", freq="MS")
    records = []

    for date in dates:
        year = date.year
        month = date.month
        t = (year - 2019) * 12 + month

        # Fed Funds Rate: ~2.4% in 2019, ~0.1% in 2020-2021, ramp to ~5.3% by end 2023
        if year <= 2020 and month >= 3:
            ff = max(0.05, 2.4 - t * 0.1)
        elif year <= 2021:
            ff = 0.08
        elif year == 2022:
            ff = 0.08 + (month / 12) * 4.0
        else:
            ff = 4.5 + (month / 12) * 0.8
        ff = round(max(0.05, ff), 2)

        # CPI (index, ~255 in 2019, ~307 by 2023)
        cpi = round(255 + t * 1.1 + (3.0 if year >= 2022 else 0) * month / 12, 1)

        # Industrial Production (index, ~105 in 2019, dip in 2020, recovery)
        ip_base = 105
        if year == 2020 and 3 <= month <= 6:
            ip = ip_base - 15 + (month - 3) * 3
        elif year == 2020:
            ip = ip_base - 3
        else:
            ip = ip_base + (t - 12) * 0.15
        ip = round(max(85, ip), 1)

        # HY Credit Spread (%, ~3.5% normal, spike to ~10% in COVID, ~4.5% in 2022)
        if year == 2020 and 3 <= month <= 5:
            hy = 3.5 + (10 - 3.5) * np.exp(-((month - 3.5) ** 2) / 0.8)
        elif year == 2022:
            hy = 4.0 + month * 0.05
        else:
            hy = 3.5 + np.random.default_rng(t).normal(0, 0.2)
        hy = round(max(2.0, hy), 2)

        date_str = date.strftime("%Y-%m-%d")
        records.append(
            {"date": date_str, "value": ff, "series_id": fred_series["fed_funds_rate"]}
        )
        records.append(
            {"date": date_str, "value": cpi, "series_id": fred_series["cpi"]}
        )
        records.append(
            {
                "date": date_str,
                "value": ip,
                "series_id": fred_series["industrial_production"],
            }
        )
        records.append(
            {
                "date": date_str,
                "value": hy,
                "series_id": fred_series["credit_spread_hy"],
            }
        )

    return pd.DataFrame(records)


def fetch_macro(config: dict) -> pd.DataFrame:
    """
    Fetch all macro series defined in config.yaml.

    Tries FRED API first (requires FRED_API_KEY env var).
    Falls back to bundled approximate data for offline use.

    Parameters
    ----------
    config : dict
        Loaded config.yaml.

    Returns
    -------
    pd.DataFrame
        Columns: date, value, series_id, series_name
    """
    fred_series = config["data"]["fred_series"]
    api_key = os.environ.get("FRED_API_KEY", "")

    frames = []

    if api_key:
        logger.info("FRED_API_KEY found — fetching live macro data...")
        for name, series_id in fred_series.items():
            try:
                df = _fetch_fred_series(series_id, api_key)
                df["series_name"] = name
                frames.append(df)
                logger.info(f"  Fetched {series_id} ({name}): {len(df)} observations")
            except Exception as e:
                logger.warning(f"  Failed to fetch {series_id}: {e}")
    else:
        logger.info("No FRED_API_KEY set — using bundled approximate macro data.")

    if not frames:
        logger.info("Generating bundled macro data as fallback...")
        df = _generate_bundled_macro_data(fred_series)
        name_map = {v: k for k, v in fred_series.items()}
        df["series_name"] = df["series_id"].map(name_map)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values(["series_id", "date"]).reset_index(drop=True)

    # Save to disk
    out_dir = Path(config["data"]["macro_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "macro_series.csv"
    combined.to_csv(out_path, index=False)
    logger.info(f"Saved {len(combined)} macro observations → {out_path}")

    return combined


def run(config_path: str = "config.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    logging.basicConfig(
        level=config["logging"]["level"],
        format=config["logging"]["format"],
    )
    return fetch_macro(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
