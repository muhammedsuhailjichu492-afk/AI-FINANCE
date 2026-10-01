
import os
import shutil
import datetime
from typing import Optional, List

from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .database import init_db, get_db, Transaction, Document
from . import ocr_processor, transaction_parser, anomaly_detector
from . import budget_analyzer, rag_engine, llm_agent, report_generator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="AI Finance & Expense Intelligence Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()


# ---------- helpers ----------

def _txn_to_dict(t: Transaction) -> dict:
    return {
        "id": t.id,
        "date": t.date.isoformat() if t.date else None,
        "vendor": t.vendor,
        "description": t.description,
        "amount": t.amount,
        "category": t.category,
        "source": t.source,
        "is_anomaly": t.is_anomaly,
        "anomaly_score": t.anomaly_score,
        "anomaly_reason": t.anomaly_reason,
        "document_path": t.document_path,
    }


def _index_for_rag(t: Transaction):
    text = f"{t.date.date() if t.date else ''} | {t.vendor} | {t.category} | ${t.amount:.2f} | {t.description or ''}"
    rag_engine.index_transaction(t.id, text, {
        "category": t.category, "vendor": t.vendor or "", "amount": t.amount,
    })


# ---------- document upload (OCR) ----------

@app.post("/api/upload/document")
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".pdf"):
        raise HTTPException(400, "Unsupported file type. Use an image or PDF invoice/receipt.")

    save_path = os.path.join(UPLOAD_DIR, f"{datetime.datetime.utcnow().timestamp():.0f}_{file.filename}")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        parsed = ocr_processor.parse_document(save_path)
    except Exception as e:
        raise HTTPException(500, f"OCR processing failed: {e}")

    if parsed["amount"] is None:
        raise HTTPException(
            422,
            "Could not detect an amount on this document. Try a clearer scan, "
            "or add the transaction manually.",
        )

    category = transaction_parser.categorize(
        parsed["vendor"], parsed["description"], llm_fallback=llm_agent.llm_categorize
    )

    txn = Transaction(
        date=parsed["date"], vendor=parsed["vendor"], description=parsed["description"],
        amount=parsed["amount"], category=category, source="ocr",
        raw_text=parsed["raw_text"], document_path=save_path,
    )
    db.add(txn)
    db.commit()
    db.refresh(txn)

    doc = Document(filename=file.filename, filepath=save_path,
                    extracted_text=parsed["raw_text"], transaction_id=txn.id)
    db.add(doc)
    db.commit()

    _index_for_rag(txn)
    return _txn_to_dict(txn)


# ---------- CSV import (real bank/expense exports) ----------

@app.post("/api/upload/csv")
async def upload_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    content = await file.read()
    try:
        records = transaction_parser.parse_csv_bytes(content)
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    if not records:
        raise HTTPException(422, "No valid transaction rows found in this CSV.")

    inserted = []
    for r in records:
        category = transaction_parser.categorize(
            r["vendor"], r["description"], llm_fallback=llm_agent.llm_categorize
        )
        txn = Transaction(
            date=r["date"], vendor=r["vendor"], description=r["description"],
            amount=r["amount"], category=category, source="csv",
        )
        db.add(txn)
        db.commit()
        db.refresh(txn)
        _index_for_rag(txn)
        inserted.append(_txn_to_dict(txn))

    return {"inserted_count": len(inserted), "transactions": inserted}


# ---------- transactions CRUD / query ----------

@app.get("/api/transactions")
def list_transactions(category: Optional[str] = None, anomaly_only: bool = False,
                       db: Session = Depends(get_db)):
    q = db.query(Transaction)
    if category:
        q = q.filter(Transaction.category == category)
    if anomaly_only:
        q = q.filter(Transaction.is_anomaly == True)  # noqa: E712
    txns = q.order_by(Transaction.date.desc()).all()
    return [_txn_to_dict(t) for t in txns]


@app.post("/api/transactions/manual")
def add_manual_transaction(
    vendor: str = Body(...), amount: float = Body(...),
    description: str = Body(""), category: Optional[str] = Body(None),
    date: Optional[str] = Body(None), db: Session = Depends(get_db),
):
    cat = category or transaction_parser.categorize(vendor, description, llm_agent.llm_categorize)
    dt = datetime.datetime.fromisoformat(date) if date else datetime.datetime.utcnow()
    txn = Transaction(date=dt, vendor=vendor, description=description,
                       amount=amount, category=cat, source="manual")
    db.add(txn)
    db.commit()
    db.refresh(txn)
    _index_for_rag(txn)
    return _txn_to_dict(txn)


@app.delete("/api/transactions/{txn_id}")
def delete_transaction(txn_id: int, db: Session = Depends(get_db)):
    t = db.query(Transaction).get(txn_id)
    if not t:
        raise HTTPException(404, "Transaction not found")
    db.delete(t)
    db.commit()
    return {"deleted": txn_id}


# ---------- analytics ----------

@app.get("/api/analytics/summary")
def analytics_summary(db: Session = Depends(get_db)):
    txns = db.query(Transaction).all()
    total = sum(t.amount for t in txns)
    by_category = {}
    by_month = {}
    for t in txns:
        by_category[t.category] = by_category.get(t.category, 0) + t.amount
        if t.date:
            key = t.date.strftime("%Y-%m")
            by_month[key] = by_month.get(key, 0) + t.amount
    return {
        "total_spend": round(total, 2),
        "total_transactions": len(txns),
        "anomaly_count": sum(1 for t in txns if t.is_anomaly),
        "by_category": [{"category": k, "total": round(v, 2)} for k, v in
                         sorted(by_category.items(), key=lambda x: -x[1])],
        "by_month": [{"month": k, "total": round(v, 2)} for k, v in sorted(by_month.items())],
    }


# ---------- anomaly detection ----------

@app.post("/api/anomalies/detect")
def detect_anomalies(db: Session = Depends(get_db)):
    flagged = anomaly_detector.run_anomaly_detection(db)
    return {"flagged_count": len(flagged), "anomalies": flagged}


@app.get("/api/anomalies")
def get_anomalies(db: Session = Depends(get_db)):
    txns = db.query(Transaction).filter(Transaction.is_anomaly == True).all()  # noqa: E712
    return [_txn_to_dict(t) for t in txns]


# ---------- budgets ----------

@app.post("/api/budget")
def set_budget(category: str = Body(...), monthly_limit: float = Body(...),
               db: Session = Depends(get_db)):
    b = budget_analyzer.set_budget(db, category, monthly_limit)
    return {"category": b.category, "monthly_limit": b.monthly_limit}


@app.get("/api/budget/analysis")
def budget_analysis(year: Optional[int] = None, month: Optional[int] = None,
                     db: Session = Depends(get_db)):
    return budget_analyzer.analyze_budgets(db, year=year, month=month)


# ---------- AI agent chat (RAG + tools) ----------

@app.post("/api/chat")
def chat(query: str = Body(..., embed=True), db: Session = Depends(get_db)):
    result = llm_agent.answer_query(db, query)
    return result


@app.get("/api/ollama/status")
def ollama_status():
    return {"available": rag_engine.is_ollama_available()}


# ---------- report ----------

@app.get("/api/report")
def get_report(db: Session = Depends(get_db)):
    return report_generator.generate_report(db)


@app.get("/api/report/markdown")
def get_report_markdown(db: Session = Depends(get_db)):
    report = report_generator.generate_report(db)
    md = report_generator.report_to_markdown(report)
    return PlainTextResponse(md, media_type="text/markdown")


# ---------- frontend ----------

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
