"""
sec_scraper.py — Financial data collection from SEC EDGAR and Yahoo Finance.

Sources:
  - SEC EDGAR XBRL API (https://data.sec.gov/api/xbrl/companyfacts/)
    Provides structured financial statement data (10-K filings).
  - yfinance for market cap, share price, and supplemental market data.

Economic context:
  The XBRL API returns standardized US-GAAP tagged values, making it the
  most reliable source for balance sheet, income statement, and cash flow
  data across public firms. yfinance fills market-data gaps (market cap
  required for Altman Z-Score X4 component).
"""

import logging
import time
from pathlib import Path
from typing import Optional

import requests
import pandas as pd
from utils.config import load_config

logger = logging.getLogger(__name__)


class SECEdgarScraper:
    """
    Fetches and normalizes financial statement data from SEC EDGAR XBRL API.

    The EDGAR XBRL API returns all reported values for a given US-GAAP concept
    across all filings for a company. We extract annual (10-K) values and
    align them to fiscal year ends.

    Parameters
    ----------
    config : dict
        Loaded config.yaml dictionary.
    """

    # Core US-GAAP XBRL concepts we need for ratio computation.
    # Keys are our internal names; values are lists of XBRL tags to try in order
    # (companies use different tags for equivalent concepts).
    XBRL_CONCEPTS = {
        "total_assets": [
            "us-gaap/Assets",
        ],
        "current_assets": [
            "us-gaap/AssetsCurrent",
        ],
        "current_liabilities": [
            "us-gaap/LiabilitiesCurrent",
        ],
        "total_liabilities": [
            "us-gaap/Liabilities",
        ],
        "retained_earnings": [
            "us-gaap/RetainedEarningsAccumulatedDeficit",
        ],
        "ebit": [
            "us-gaap/OperatingIncomeLoss",  # best proxy; EBIT = operating income for most firms
        ],
        "revenue": [
            "us-gaap/RevenueFromContractWithCustomerExcludingAssessedTax",
            "us-gaap/Revenues",
            "us-gaap/SalesRevenueNet",
        ],
        "interest_expense": [
            "us-gaap/InterestExpense",
            "us-gaap/InterestAndDebtExpense",
        ],
        "accounts_payable": [
            "us-gaap/AccountsPayableCurrent",
        ],
        "accounts_receivable": [
            "us-gaap/AccountsReceivableNetCurrent",
        ],
        "cogs": [
            "us-gaap/CostOfGoodsAndServicesSold",
            "us-gaap/CostOfRevenue",
        ],
        "long_term_debt": [
            "us-gaap/LongTermDebtNoncurrent",
            "us-gaap/LongTermDebt",
        ],
        "stockholders_equity": [
            "us-gaap/StockholdersEquity",
            "us-gaap/StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ],
        "shares_outstanding": [
            "us-gaap/CommonStockSharesOutstanding",
        ],
    }

    def __init__(self, config: dict):
        self.config = config
        self.base_url = config["data"]["sec_base_url"]
        self.user_agent = config["data"]["sec_user_agent"]
        self.output_dir = Path(config["data"]["financials_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.years = config["data"]["years"]
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept": "application/json",
            }
        )

    def _get_company_facts(self, cik: str) -> Optional[dict]:
        """
        Fetch all XBRL facts for a company from SEC EDGAR.

        Parameters
        ----------
        cik : str
            SEC CIK number (will be zero-padded to 10 digits).

        Returns
        -------
        dict or None
            Raw EDGAR companyfacts JSON, or None on failure.
        """
        cik_padded = str(cik).zfill(10)
        url = f"{self.base_url}/api/xbrl/companyfacts/CIK{cik_padded}.json"
        for attempt in range(3):
            try:
                resp = self.session.get(url, timeout=30)
                resp.raise_for_status()
                return resp.json()
            except requests.HTTPError as e:
                logger.warning(f"HTTP {e.response.status_code} for CIK {cik}: {e}")
                if e.response.status_code == 404:
                    return None  # firm not found; no retry
                time.sleep(2**attempt)
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for CIK {cik}: {e}")
                time.sleep(2**attempt)
        logger.error(f"All attempts failed for CIK {cik}")
        return None

    def _extract_annual_series(self, facts: dict, concept_path: str) -> pd.Series:
        """
        Extract annual 10-K values for a single XBRL concept.

        Parameters
        ----------
        facts : dict
            Raw EDGAR companyfacts JSON.
        concept_path : str
            Taxonomy/concept string e.g. "us-gaap/Assets".

        Returns
        -------
        pd.Series
            Index = fiscal year (int), values = reported amounts (USD).
        """
        try:
            taxonomy, concept = concept_path.split("/", 1)
            units = facts["facts"][taxonomy][concept]["units"]
            # USD is the standard unit for financial figures
            entries = units.get("USD", units.get("shares", []))
        except (KeyError, TypeError):
            return pd.Series(dtype=float)

        records = []
        for entry in entries:
            # Filter for annual 10-K filings only
            if entry.get("form") not in ("10-K", "10-K/A"):
                continue
            if entry.get("fp") != "FY":
                continue
            end_date = pd.to_datetime(entry.get("end", ""))
            if pd.isnull(end_date):
                continue
            year = end_date.year
            records.append({"year": year, "val": entry["val"]})

        if not records:
            return pd.Series(dtype=float)

        df = pd.DataFrame(records)
        # If multiple filings for same year (amendments), take the last filed
        df = df.drop_duplicates(subset="year", keep="last")
        return df.set_index("year")["val"]

    def _get_market_cap(self, ticker: str, year: int) -> float:
        """
        Retrieve approximate year-end market cap from yfinance.

        Market cap is required for Altman Z-Score X4 = Market Cap / Total Liabilities.
        We use the December 31 closing price × shares outstanding as a proxy
        for fiscal year-end market cap.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol.
        year : int
            Fiscal year.

        Returns
        -------
        float
            Market cap in USD, or NaN if unavailable.
        """
        try:
            import yfinance as yf

            start = f"{year}-12-15"
            end = f"{year + 1}-01-15"
            data = yf.download(
                ticker, start=start, end=end, progress=False, auto_adjust=True
            )
            if data.empty:
                return float("nan")
            close = float(data["Close"].iloc[-1])
            info = yf.Ticker(ticker).fast_info
            shares = getattr(info, "shares", None) or getattr(
                info, "shares_outstanding", None
            )
            if shares is None:
                return float("nan")
            return close * shares
        except Exception as e:
            logger.warning(f"Could not get market cap for {ticker} ({year}): {e}")
            return float("nan")

    def fetch_firm(self, ticker: str, cik: str, name: str) -> pd.DataFrame:
        """
        Fetch, normalize, and return annual financial data for one firm.

        Pulls XBRL data from SEC EDGAR and market cap from yfinance.
        Missing values are left as NaN and flagged in the data quality report.

        Parameters
        ----------
        ticker : str
            Stock ticker (e.g., "TM").
        cik : str
            SEC CIK number.
        name : str
            Human-readable firm name.

        Returns
        -------
        pd.DataFrame
            One row per year with standardized financial columns.
            Saved to data/financials/{ticker}.parquet.
        """
        logger.info(f"Fetching {name} ({ticker}, CIK {cik})")
        facts = self._get_company_facts(cik)
        if facts is None:
            logger.error(f"No EDGAR data for {name}. Returning empty frame.")
            return pd.DataFrame()

        # Build one dict per concept
        concept_series: dict[str, pd.Series] = {}
        for field, tags in self.XBRL_CONCEPTS.items():
            for tag in tags:
                series = self._extract_annual_series(facts, tag)
                if not series.empty:
                    concept_series[field] = series
                    break
            else:
                logger.warning(f"{ticker}: No data for '{field}' (tried {tags})")
                concept_series[field] = pd.Series(dtype=float)

        # Align to target years
        rows = []
        for year in self.years:
            row = {"ticker": ticker, "name": name, "year": year}
            for field, series in concept_series.items():
                row[field] = series.get(year, float("nan"))
            # Market cap from yfinance
            row["market_cap"] = self._get_market_cap(ticker, year)
            rows.append(row)

        df = pd.DataFrame(rows)
        out_path = self.output_dir / f"{ticker}.parquet"
        df.to_parquet(out_path, index=False)
        logger.info(f"Saved {ticker} → {out_path} ({len(df)} rows)")
        return df

    def fetch_all(self) -> pd.DataFrame:
        """
        Fetch financial data for all firms defined in config.yaml.

        Returns
        -------
        pd.DataFrame
            Combined dataframe for all firms, all years.
        """
        cfg = self.config
        all_firms = []
        for tier in ("tier1_buyers", "tier2_suppliers", "tier3_suppliers"):
            all_firms.extend(cfg["firms"].get(tier, []))

        frames = []
        seen_tickers = set()
        for firm in all_firms:
            ticker = firm.get("ticker")
            cik = firm.get("cik")
            name = firm.get("name", ticker)
            if not ticker or not cik:
                logger.warning(f"Skipping malformed firm entry: {firm}")
                continue
            if ticker in seen_tickers:
                continue
            seen_tickers.add(ticker)
            df = self.fetch_firm(ticker, cik, name)
            if not df.empty:
                frames.append(df)
            time.sleep(0.2)  # SEC rate-limit courtesy pause

        if not frames:
            logger.error("No financial data retrieved for any firm.")
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined.to_parquet(self.output_dir / "all_firms.parquet", index=False)
        return combined


def run(config_path: str = "config.yaml") -> pd.DataFrame:
    """
    Entry point: fetch all financial data.

    Parameters
    ----------
    config_path : str
        Path to config.yaml.

    Returns
    -------
    pd.DataFrame
        Combined financial data for all firms.
    """
    config = load_config(config_path)
    logging.basicConfig(
        level=config["logging"]["level"],
        format=config["logging"]["format"],
    )
    scraper = SECEdgarScraper(config)
    return scraper.fetch_all()


if __name__ == "__main__":
    run()
