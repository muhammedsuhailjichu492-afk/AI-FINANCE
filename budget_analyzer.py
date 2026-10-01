
import datetime
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func

from .database import Transaction, Budget


def analyze_budgets(db: Session, year: int = None, month: int = None) -> List[Dict[str, Any]]:
    now = datetime.datetime.utcnow()
    year = year or now.year
    month = month or now.month

    spend_rows = (
        db.query(Transaction.category, func.sum(Transaction.amount))
        .filter(func.strftime("%Y", Transaction.date) == str(year))
        .filter(func.strftime("%m", Transaction.date) == f"{month:02d}")
        .group_by(Transaction.category)
        .all()
    )
    spend_by_cat = {cat: float(total) for cat, total in spend_rows}

    budgets = {b.category: b.monthly_limit for b in db.query(Budget).all()}

    all_categories = set(spend_by_cat) | set(budgets)
    results = []
    for cat in sorted(all_categories):
        spent = spend_by_cat.get(cat, 0.0)
        limit = budgets.get(cat)
        if limit is None:
            status = "no_budget_set"
            pct = None
        else:
            pct = (spent / limit * 100) if limit > 0 else None
            if pct is None:
                status = "unknown"
            elif pct > 100:
                status = "over_budget"
            elif pct > 85:
                status = "near_limit"
            else:
                status = "on_track"
        results.append({
            "category": cat,
            "spent": round(spent, 2),
            "budget": limit,
            "percent_used": round(pct, 1) if pct is not None else None,
            "status": status,
        })
    return results


def set_budget(db: Session, category: str, monthly_limit: float) -> Budget:
    existing = db.query(Budget).filter(Budget.category == category).first()
    if existing:
        existing.monthly_limit = monthly_limit
        db.commit()
        db.refresh(existing)
        return existing
    b = Budget(category=category, monthly_limit=monthly_limit)
    db.add(b)
    db.commit()
    db.refresh(b)
    return b
