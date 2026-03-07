"""
main.py — Financial Supply Chain Risk Platform — Pipeline Orchestrator.

Usage:
    python main.py seed         Seed demo data (offline, no API)
    python main.py fetch        Fetch financial data from SEC EDGAR + yfinance
    python main.py macro        Fetch macroeconomic data from FRED
    python main.py scores       Compute financial ratios and Z-scores
    python main.py edges        Build supply chain edges
    python main.py quality      Generate data quality report
    python main.py dashboard    Launch Streamlit dashboard
    python main.py all          Run full pipeline (fetch + edges + macro + scores + quality)
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def cmd_seed(_args):
    from scripts.seed_demo_data import seed_all
    import yaml

    with open(ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)
    seed_all(config)


def cmd_fetch(_args):
    from data_engineering.sec_scraper import run

    run()


def cmd_macro(_args):
    from data_engineering.macro_fetcher import run

    run()


def cmd_edges(_args):
    from data_engineering.supply_chain_builder import run

    run()


def cmd_scores(_args):
    from data_engineering.ratio_calculator import run

    run()


def cmd_quality(_args):
    from data_engineering.data_quality_report import run

    run()


def cmd_dashboard(_args):
    import subprocess

    subprocess.run(["streamlit", "run", str(ROOT / "dashboard" / "app.py")])


def cmd_all(_args):
    print("==> Running full pipeline...")
    cmd_fetch(_args)
    cmd_edges(_args)
    cmd_macro(_args)
    cmd_scores(_args)
    cmd_quality(_args)
    print("==> Full pipeline complete. Run: python main.py dashboard")


def main():
    parser = argparse.ArgumentParser(
        description="Financial Supply Chain Risk Platform — Pipeline Orchestrator"
    )
    sub = parser.add_subparsers(dest="command", help="Pipeline command")

    sub.add_parser("seed", help="Seed demo data (offline)")
    sub.add_parser("fetch", help="Fetch financial data from SEC EDGAR + yfinance")
    sub.add_parser("macro", help="Fetch macroeconomic data from FRED")
    sub.add_parser("edges", help="Build supply chain edges")
    sub.add_parser("scores", help="Compute financial ratios and Z-scores")
    sub.add_parser("quality", help="Generate data quality report")
    sub.add_parser("dashboard", help="Launch Streamlit dashboard")
    sub.add_parser("all", help="Run full pipeline end-to-end")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    commands = {
        "seed": cmd_seed,
        "fetch": cmd_fetch,
        "macro": cmd_macro,
        "edges": cmd_edges,
        "scores": cmd_scores,
        "quality": cmd_quality,
        "dashboard": cmd_dashboard,
        "all": cmd_all,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
