import requests
from bs4 import BeautifulSoup
import openpyxl

URL = "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm"
HEADERS = {"User-Agent": "Financial Research Tool contact@example.com"}

# ── 1. Fetch and parse the page ───────────────────────────────────
print("Fetching 10-K... (this may take a minute)")
response = requests.get(URL, headers=HEADERS, timeout=90)
soup = BeautifulSoup(response.content, "html.parser")
tables = soup.find_all("table")
print(f"Found {len(tables)} tables")

# ── 2. Helper: search all tables for a label, return first number ─
def find_value(label_pattern, col=1):
    import re
    pat = re.compile(label_pattern, re.IGNORECASE)
    for table in tables:
        for row in table.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in row.find_all(["td", "th"])]
            if cells and pat.search(cells[0]):
                for cell in cells[col:]:
                    text = cell.replace(",", "").replace("$", "").replace("\xa0", "").strip()
                    negative = text.startswith("(") and text.endswith(")")
                    text = text.strip("()")
                    try:
                        val = float(text)
                        return -val if negative else val
                    except ValueError:
                        continue
    return None

# ── 3. Extract the 5 components ───────────────────────────────────
print("Extracting values...")

working_capital   = find_value(r"total current assets") - find_value(r"total current liabilities")
total_assets      = find_value(r"^total assets$")
retained_earnings = find_value(r"retained earnings")
ebit              = find_value(r"income from operations|operating income")
total_liabilities = find_value(r"^total liabilities$")
revenue           = find_value(r"^revenue$|^net revenue")

price = float(input("Enter current NVDA share price ($): "))
market_cap = price * 24300   # 24.3 billion shares x price = market cap in $M

# ── 4. Altman Z-Score ─────────────────────────────────────────────
X1 = working_capital   / total_assets
X2 = retained_earnings / total_assets
X3 = ebit              / total_assets
X4 = market_cap        / total_liabilities
X5 = revenue           / total_assets

Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

# ── 5. Write to Excel ─────────────────────────────────────────────
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Altman Z-Score"

ws.append(["Component", "Formula", "Value"])
ws.append(["Working Capital ($M)",   "Current Assets - Current Liabilities", working_capital])
ws.append(["Total Assets ($M)",      "From Balance Sheet",                   total_assets])
ws.append(["Retained Earnings ($M)", "From Balance Sheet",                   retained_earnings])
ws.append(["EBIT ($M)",              "Operating Income",                     ebit])
ws.append(["Total Liabilities ($M)", "From Balance Sheet",                   total_liabilities])
ws.append(["Revenue ($M)",           "From Income Statement",                revenue])
ws.append(["Market Cap ($M)",        f"24.3B shares x ${price}",             market_cap])
ws.append([])
ws.append(["X1 (Working Capital / Total Assets)",       "", X1])
ws.append(["X2 (Retained Earnings / Total Assets)",     "", X2])
ws.append(["X3 (EBIT / Total Assets)",                  "", X3])
ws.append(["X4 (Market Cap / Total Liabilities)",       "", X4])
ws.append(["X5 (Revenue / Total Assets)",               "", X5])
ws.append([])
ws.append(["Altman Z-Score", "1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5", Z])

wb.save("nvda_altman_zscore2026.xlsx")
print(f"\nAltman Z-Score: {Z:.2f}")
print("Saved: nvda_altman_zscore2026.xlsx")
