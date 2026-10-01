
import datetime
from sqlalchemy.orm import Session
from sqlalchemy import func

from .database import Transaction
from . import anomaly_detector, budget_analyzer, llm_agent


def _risk_score(total_txns: int, anomaly_count: int, over_budget_count: int, total_categories: int) -> int:
    if total_txns == 0:
        return 0
    anomaly_ratio = anomaly_count / total_txns
    budget_ratio = (over_budget_count / total_categories) if total_categories else 0
    score = min(100, round((anomaly_ratio * 60 + budget_ratio * 40) * 100))
    return score


def generate_report(db: Session) -> dict:
    transactions = db.query(Transaction).all()
    total_spend = sum(t.amount for t in transactions)
    total_txns = len(transactions)

    by_cat = (
        db.query(Transaction.category, func.sum(Transaction.amount), func.count(Transaction.id))
        .group_by(Transaction.category)
        .order_by(func.sum(Transaction.amount).desc())
        .all()
    )

    anomalies = anomaly_detector.run_anomaly_detection(db)
    budgets = budget_analyzer.analyze_budgets(db)
    over_budget = [b for b in budgets if b["status"] == "over_budget"]

    risk = _risk_score(total_txns, len(anomalies), len(over_budget), len(budgets))

    context_lines = [
        f"Total transactions: {total_txns}",
        f"Total spend: ${total_spend:,.2f}",
        "Category breakdown: " + "; ".join(
            f"{cat} ${amt:,.2f} ({cnt} txns)" for cat, amt, cnt in by_cat
        ),
        f"Anomalies flagged: {len(anomalies)}",
        f"Categories over budget: {len(over_budget)} of {len(budgets)}",
        f"Computed risk score: {risk}/100",
    ]
    context = "\n".join(context_lines)

    exec_summary = llm_agent.generate_executive_summary(context)

    report = {
        "generated_at": datetime.datetime.utcnow().isoformat(),
        "executive_summary": exec_summary,
        "total_spend": round(total_spend, 2),
        "total_transactions": total_txns,
        "risk_score": risk,
        "category_breakdown": [
            {"category": cat, "total": round(float(amt), 2), "count": cnt}
            for cat, amt, cnt in by_cat
        ],
        "top_anomalies": anomalies[:10],
        "budget_status": budgets,
    }
    return report


def report_to_markdown(report: dict) -> str:
    lines = [
        "# AI Finance & Expense Intelligence Report",
        f"_Generated: {report['generated_at']}_",
        "",
        "## Executive Summary",
        report["executive_summary"],
        "",
        f"**Total Spend:** ${report['total_spend']:,.2f}  ",
        f"**Total Transactions:** {report['total_transactions']}  ",
        f"**Risk Score:** {report['risk_score']}/100",
        "",
        "## Category Breakdown",
        "| Category | Total | Transactions |",
        "|---|---|---|",
    ]
    for c in report["category_breakdown"]:
        lines.append(f"| {c['category']} | ${c['total']:,.2f} | {c['count']} |")

    lines += ["", "## Top Anomalies", "| ID | Vendor | Amount | Category | Reason |",
              "|---|---|---|---|---|"]
    for a in report["top_anomalies"]:
        lines.append(
            f"| {a['id']} | {a['vendor']} | ${a['amount']:,.2f} | {a['category']} | {a['reason']} |"
        )

    lines += ["", "## Budget Status", "| Category | Spent | Budget | % Used | Status |",
              "|---|---|---|---|---|"]
    for b in report["budget_status"]:
        budget_str = f"${b['budget']:,.2f}" if b["budget"] is not None else "—"
        pct_str = f"{b['percent_used']}%" if b["percent_used"] is not None else "—"
        lines.append(f"| {b['category']} | ${b['spent']:,.2f} | {budget_str} | {pct_str} | {b['status']} |")

    return "\n".join(lines)
