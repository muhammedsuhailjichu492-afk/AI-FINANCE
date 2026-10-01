
import os
import json
import requests
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from sqlalchemy import func

from .database import Transaction
from . import rag_engine, anomaly_detector, budget_analyzer

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "llama3")

SYSTEM_PROMPT = (
    "You are a financial analysis assistant embedded in an AI Finance & Expense "
    "Intelligence platform used by corporate finance and procurement teams. "
    "You answer questions ONLY using the CONTEXT provided below, which is pulled "
    "from the organization's real transaction records, anomaly detections, and "
    "budget data. Be precise, cite specific numbers from the context, and if the "
    "context does not contain the answer, say so plainly instead of guessing. "
    "Keep answers concise and business-appropriate."
)


def ollama_generate(prompt: str, system: str = None) -> str:
    payload = {
        "model": CHAT_MODEL,
        "prompt": prompt,
        "system": system or "",
        "stream": False,
    }
    resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def llm_categorize(vendor: str, description: str) -> str:
    """Fallback categorizer used by transaction_parser when no keyword rule matches."""
    prompt = (
        "Classify this business expense into exactly one category from this list: "
        "Travel, Meals & Entertainment, Office Supplies, Software & IT, Utilities, "
        "Procurement / Raw Materials, Rent & Facilities, Marketing, Payroll, "
        "Insurance, Logistics & Shipping, Professional Services, Other.\n"
        f"Vendor: {vendor}\nDescription: {description}\n"
        "Respond with ONLY the category name, nothing else."
    )
    return ollama_generate(prompt)


def _sql_summary_tool(db: Session) -> str:
    total = db.query(func.sum(Transaction.amount)).scalar() or 0
    count = db.query(func.count(Transaction.id)).scalar() or 0
    by_cat = (
        db.query(Transaction.category, func.sum(Transaction.amount))
        .group_by(Transaction.category)
        .order_by(func.sum(Transaction.amount).desc())
        .all()
    )
    lines = [f"Total spend across {count} transactions: ${total:,.2f}", "By category:"]
    for cat, amt in by_cat:
        lines.append(f"  - {cat}: ${amt:,.2f}")
    return "\n".join(lines)


def _anomaly_tool(db: Session) -> str:
    flagged = anomaly_detector.run_anomaly_detection(db)
    if not flagged:
        return "No anomalies currently flagged (or not enough transaction history yet)."
    lines = ["Flagged anomalous transactions:"]
    for f in flagged[:15]:
        lines.append(
            f"  - #{f['id']} {f['vendor']} ${f['amount']:,.2f} ({f['category']}) — {f['reason']}"
        )
    return "\n".join(lines)


def _budget_tool(db: Session) -> str:
    results = budget_analyzer.analyze_budgets(db)
    if not results:
        return "No budget data available yet."
    lines = ["Budget vs actual (current month):"]
    for r in results:
        budget_str = f"${r['budget']:,.2f}" if r['budget'] is not None else "not set"
        pct_str = f"{r['percent_used']}%" if r['percent_used'] is not None else "n/a"
        lines.append(
            f"  - {r['category']}: spent ${r['spent']:,.2f} / budget {budget_str} "
            f"({pct_str}) — {r['status']}"
        )
    return "\n".join(lines)


def _route(query: str) -> List[str]:
    """Decide which tools are relevant to the question."""
    q = query.lower()
    tools = []
    if any(k in q for k in ["anomaly", "anomalies", "unusual", "suspicious", "fraud", "risk"]):
        tools.append("anomaly")
    if any(k in q for k in ["budget", "over budget", "limit", "overspend", "under budget"]):
        tools.append("budget")
    if any(k in q for k in ["total", "spend", "spending", "breakdown", "how much", "summary"]):
        tools.append("sql")
    if not tools:
        tools = ["rag"]  # default to retrieval over real transaction text
    return tools


def answer_query(db: Session, query: str) -> Dict[str, Any]:
    """Main agent entrypoint used by the /api/chat endpoint."""
    tools_used = _route(query)
    context_parts = []

    if "sql" in tools_used:
        context_parts.append("[SQL ANALYTICS]\n" + _sql_summary_tool(db))
    if "anomaly" in tools_used:
        context_parts.append("[ANOMALY DETECTION]\n" + _anomaly_tool(db))
    if "budget" in tools_used:
        context_parts.append("[BUDGET ANALYSIS]\n" + _budget_tool(db))

    # Always pull relevant real records via RAG for grounding
    similar = rag_engine.query_similar(query, n_results=5)
    if similar:
        rag_lines = ["[RELEVANT RECORDS]"]
        for s in similar:
            rag_lines.append(f"  - {s['text']}")
        context_parts.append("\n".join(rag_lines))
        if "rag" not in tools_used:
            tools_used.append("rag")

    context = "\n\n".join(context_parts) if context_parts else "No data available yet."

    if not rag_engine.is_ollama_available():
        return {
            "answer": (
                "Ollama isn't reachable at the moment, so I can't generate a "
                "natural-language answer, but here is the raw context I would "
                f"have reasoned over:\n\n{context}"
            ),
            "tools_used": tools_used,
            "context": context,
        }

    prompt = f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nANSWER:"
    try:
        answer = ollama_generate(prompt, system=SYSTEM_PROMPT)
    except requests.RequestException as e:
        answer = f"Error calling local LLM: {e}\n\nRaw context:\n{context}"

    return {"answer": answer, "tools_used": tools_used, "context": context}


def generate_executive_summary(context: str) -> str:
    if not rag_engine.is_ollama_available():
        return "Ollama not available — executive summary could not be generated."
    prompt = (
        "Write a 4-6 sentence executive summary for a corporate finance report "
        "based on the following real data. Be specific with numbers, mention "
        "the biggest risk or anomaly, and note overall budget health.\n\n"
        f"{context}"
    )
    try:
        return ollama_generate(prompt, system=SYSTEM_PROMPT)
    except requests.RequestException as e:
        return f"Could not generate summary: {e}"
