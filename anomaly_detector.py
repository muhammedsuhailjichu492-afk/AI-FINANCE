
from typing import List, Dict, Any
import numpy as np
from sklearn.ensemble import IsolationForest
from sqlalchemy.orm import Session

from .database import Transaction


def run_anomaly_detection(db: Session, z_threshold: float = 2.5,
                           contamination: float = 0.05) -> List[Dict[str, Any]]:
    transactions = db.query(Transaction).all()
    if len(transactions) < 5:
        return []  # not enough real data yet to detect meaningful outliers

    categories = sorted({t.category for t in transactions})
    cat_index = {c: i for i, c in enumerate(categories)}

    X = np.array([
        [t.amount, t.date.weekday() if t.date else 0, cat_index[t.category]]
        for t in transactions
    ])

    # 1. Isolation Forest (global structural outliers)
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso_labels = iso.fit_predict(X)
    iso_scores = -iso.score_samples(X)  # higher = more anomalous

    # 2. Per-category z-score (spending pattern outliers within their own category)
    amounts_by_cat = {}
    for t in transactions:
        amounts_by_cat.setdefault(t.category, []).append(t.amount)

    stats_by_cat = {}
    for cat, amts in amounts_by_cat.items():
        arr = np.array(amts)
        stats_by_cat[cat] = (arr.mean(), arr.std() if arr.std() > 0 else 1.0)

    flagged = []
    for i, t in enumerate(transactions):
        mean, std = stats_by_cat[t.category]
        z = (t.amount - mean) / std
        is_iso_outlier = iso_labels[i] == -1
        is_z_outlier = abs(z) > z_threshold

        t.anomaly_score = float(round(iso_scores[i], 4))
        t.is_anomaly = bool(is_iso_outlier or is_z_outlier)

        reasons = []
        if is_iso_outlier:
            reasons.append("unusual combination of amount/timing/category")
        if is_z_outlier:
            direction = "higher" if z > 0 else "lower"
            reasons.append(f"{abs(z):.1f}x std-dev {direction} than typical '{t.category}' spend")
        t.anomaly_reason = "; ".join(reasons) if reasons else None

        if t.is_anomaly:
            flagged.append({
                "id": t.id,
                "date": t.date.isoformat() if t.date else None,
                "vendor": t.vendor,
                "amount": t.amount,
                "category": t.category,
                "anomaly_score": t.anomaly_score,
                "reason": t.anomaly_reason,
            })

    db.commit()
    flagged.sort(key=lambda x: x["anomaly_score"], reverse=True)
    return flagged
