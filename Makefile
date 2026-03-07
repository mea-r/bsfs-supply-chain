# ============================================================
# Financial Supply Chain Risk Platform — Makefile
# ============================================================
# Commands:
#   make install        Install Python dependencies
#   make supply_chain   Build supply chain edges (no API required)
#   make data           Fetch financial data from SEC EDGAR + yfinance
#   make scores         Compute financial ratios and Z-scores
#   make quality        Generate data quality report
#   make dashboard      Launch Streamlit dashboard
#   make test           Run pytest suite
#   make all            Full pipeline end-to-end
#   make seed           Seed demo data (for offline/no-API use)
#   make clean          Remove generated data files

PYTHON := python
STREAMLIT := streamlit

.PHONY: install supply_chain data scores quality dashboard test all seed clean

install:
	pip install -r requirements.txt

supply_chain:
	@echo "==> Building supply chain edges..."
	$(PYTHON) -m data_engineering.supply_chain_builder

data:
	@echo "==> Fetching financial data from SEC EDGAR + Yahoo Finance..."
	$(PYTHON) -m data_engineering.sec_scraper
	@echo "==> Data saved to data/financials/"

scores: data
	@echo "==> Computing financial ratios and Altman Z-Scores..."
	$(PYTHON) -m data_engineering.ratio_calculator
	@echo "==> Scores saved to risk_framework/scores.csv"

quality: scores
	@echo "==> Generating data quality report..."
	$(PYTHON) -m data_engineering.data_quality_report
	@echo "==> Report saved to data_quality_report.md"

dashboard:
	@echo "==> Launching Streamlit dashboard..."
	@echo "    Open http://localhost:8501 in your browser"
	$(STREAMLIT) run dashboard/app.py

test:
	@echo "==> Running pytest suite..."
	$(PYTHON) -m pytest tests/ -v --tb=short

all: install supply_chain scores quality
	@echo "==> Full pipeline complete."
	@echo "    Run 'make dashboard' to launch the interactive dashboard."

seed:
	@echo "==> Seeding demo data (offline mode — no API calls)..."
	$(PYTHON) scripts/seed_demo_data.py
	@echo "==> Demo data ready. Run 'make dashboard' to explore."

clean:
	@echo "==> Removing generated data files..."
	rm -f data/financials/*.parquet
	rm -f data/supply_chain/edges.csv
	rm -f data/macro/macro_series.csv
	rm -f risk_framework/scores.csv
	rm -f data_quality_report.md
	@echo "==> Clean complete."
