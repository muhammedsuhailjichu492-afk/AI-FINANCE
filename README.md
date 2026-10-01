# AI Finance & Expense Intelligence Agent

An intelligent financial assistant that analyzes **real** expenses, invoices, budgets
and transactions — no mock/demo data baked in. You feed it your own scanned
invoices/receipts and/or CSV bank exports, and it runs the full pipeline:

```
Document Processing (OCR) → Transaction Analysis (categorization + SQL)
  → Anomaly Detection (Isolation Forest + z-score) → Budget Analysis → Report
```

plus a RAG-grounded AI assistant chat that answers questions using your actual data.

**Stack:** FastAPI (Python) backend + a plain HTML/CSS/JS web dashboard (no Streamlit).
LLM + embeddings run locally through **Ollama** (matches your existing local setup —
`llama3` for reasoning, `nomic-embed-text` for embeddings), SQLite for transactional
SQL analytics, ChromaDB for the vector store, scikit-learn for anomaly detection,
Tesseract for OCR.

---

## 1. Prerequisites (Windows)

1. **Python 3.10+** — you already have this.
2. **Tesseract OCR** (for reading invoices/receipts):
   ```powershell
   winget install --id UB-Mannheim.TesseractOCR
   ```
   After install, note the path (usually `C:\Program Files\Tesseract-OCR\tesseract.exe`)
   and add it to your PATH, or set it explicitly at the top of `backend/ocr_processor.py`:
   ```python
   import pytesseract
   pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
   ```
3. **Poppler** (only needed if you'll upload PDF invoices, not just images):
   ```powershell
   winget install --id oschwartz10612.Poppler
   ```
   Add its `Library\bin` folder to PATH so `pdf2image` can find `pdftoppm`.
4. **Ollama** — already running on your machine per your usual setup. Make sure these
   models are pulled:
   ```powershell
   ollama pull llama3
   ollama pull nomic-embed-text
   ```
   The app works fine without Ollama running too — OCR, SQL analytics, anomaly
   detection, and budget analysis all work standalone; only the chat assistant and
   the report's executive summary need Ollama, and both degrade gracefully
   (you still get the raw data/report) if it's not reachable.

---

## 2. Setup

Open this folder in VS Code, then in the integrated PowerShell terminal:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks the activation script, run once:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 3. Run

```powershell
uvicorn backend.main:app --reload --port 8000
```

Open **http://localhost:8000** in your browser — that's the web interface
(Dashboard, Upload Data, Transactions, Budgets, AI Assistant, Report tabs).

---

## 4. Using it with real data

- **Upload Data tab → Upload Invoice/Receipt**: upload a real photo/scan/PDF of an
  invoice or receipt. It's OCR'd, fields are extracted, it's auto-categorized, and
  it's stored.
- **Upload Data tab → Import CSV**: export a real transaction history from your
  bank/card/ERP as CSV and import it directly. Column names (date, amount,
  vendor/merchant/description) are auto-detected — see
  `sample_data/csv_format_template.csv` for the expected shape, not sample content.
- **Budgets tab**: set a real monthly budget per category (the categories in
  `backend/transaction_parser.py`'s `CATEGORY_KEYWORDS`, or any custom text — budgets
  are matched by exact category name).
- **Transactions tab → Run Anomaly Detection**: re-scores every transaction using
  Isolation Forest + per-category z-score and flags outliers.
- **AI Assistant tab**: ask things like *"What's unusual about this month's spend?"*
  or *"How much did we spend on Procurement?"* — it routes to SQL/anomaly/budget
  tools and retrieves relevant real transactions via RAG before answering.
- **Report tab**: generates a full markdown report (executive summary, category
  breakdown, top anomalies, budget status, computed risk score) from whatever real
  data is currently in the database, downloadable as `.md`.

There is no seeded/demo data — the database starts empty, and everything you see is
built from what you actually upload.

---

## 5. Project structure

```
ai-finance-agent/
├── backend/
│   ├── main.py                # FastAPI app & all REST endpoints
│   ├── database.py            # SQLAlchemy models (Transaction, Budget, Document)
│   ├── ocr_processor.py       # OCR + field extraction (Document Processing)
│   ├── transaction_parser.py  # Categorization + CSV import (Transaction Analysis)
│   ├── anomaly_detector.py    # Isolation Forest + z-score (Anomaly Detection)
│   ├── budget_analyzer.py     # Budget vs actual (Budget Analysis)
│   ├── rag_engine.py          # Ollama embeddings + ChromaDB (RAG)
│   ├── llm_agent.py           # Tool-routing agent + Ollama chat (Agents/LLM)
│   └── report_generator.py    # Risk score + markdown report (Report)
├── frontend/
│   ├── index.html             # Web dashboard (6 tabs)
│   ├── style.css
│   └── app.js                 # Fetch calls, Chart.js charts, chat UI
├── data/                      # SQLite DB, ChromaDB store, uploaded files (created at runtime)
├── sample_data/csv_format_template.csv
└── requirements.txt
```

## 6. Notes on the "Agents" piece

`backend/llm_agent.py` implements a small explicit tool-routing agent (no LangChain,
consistent with how you've built your other RAG projects): it looks at the question,
decides which of the SQL / anomaly / budget / RAG tools are relevant, pulls their
real output, and hands that as grounded context to the local Ollama model for the
final natural-language answer. You can extend `_route()` and add new `_*_tool()`
functions to add more tools (e.g. a vendor-lookup tool, a monthly-comparison tool).

## 7. Extending it

- Swap SQLite for Postgres by changing `DB_PATH`/engine URL in `database.py` (you've
  done asyncpg + Postgres before in the Resume Screening project, so this is a drop-in
  change if you want it production-grade).
- The OCR field extraction in `ocr_processor.py` is regex/heuristic-based; if you're
  processing invoices from a small set of known vendors, you'll get better accuracy
  tailoring the regex patterns to their specific layouts.
- `report_generator.py`'s risk score formula is intentionally simple
  (`60% anomaly ratio + 40% over-budget ratio`) — tune the weights once you have
  enough real historical data to calibrate against.
