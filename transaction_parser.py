
import io
import csv
import datetime
from typing import List, Dict, Any, Optional

import pandas as pd

CATEGORY_KEYWORDS = {
    "Travel": ["airlines", "airline", "uber", "lyft", "taxi", "hotel", "flight",
               "train", "railway", "car rental", "airbnb", "booking.com"],
    "Meals & Entertainment": ["restaurant", "cafe", "coffee", "starbucks", "catering",
                               "diner", "bar & grill", "food", "swiggy", "zomato"],
    "Office Supplies": ["staples", "office depot", "stationery", "paper co",
                          "printer", "ink", "supplies"],
    "Software & IT": ["aws", "azure", "google cloud", "software", "saas", "license",
                        "github", "microsoft", "adobe", "hosting", "domain", "server"],
    "Utilities": ["electricity", "water board", "gas company", "utility", "internet",
                  "broadband", "telecom", "power corporation"],
    "Procurement / Raw Materials": ["steel", "raw material", "supplier", "manufacturing",
                                      "components", "industrial supply", "wholesale",
                                      "bulk order", "factory"],
    "Rent & Facilities": ["rent", "lease", "facility", "property management", "warehouse"],
    "Marketing": ["marketing", "advertising", "ads", "campaign", "media agency"],
    "Payroll": ["payroll", "salary", "wages", "employee payment"],
    "Insurance": ["insurance", "premium payment"],
    "Logistics & Shipping": ["fedex", "ups", "dhl", "freight", "shipping", "logistics",
                               "courier"],
    "Professional Services": ["consulting", "legal", "law firm", "accounting",
                                "audit", "advisory"],
}


def rule_based_category(text: str) -> Optional[str]:
    lower = (text or "").lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                return category
    return None


def categorize(vendor: str, description: str, llm_fallback=None) -> str:
    """Categorize a transaction. Tries keyword rules, then an optional LLM
    fallback function (vendor, description) -> category string."""
    combined = f"{vendor} {description}"
    cat = rule_based_category(combined)
    if cat:
        return cat
    if llm_fallback is not None:
        try:
            llm_cat = llm_fallback(vendor, description)
            if llm_cat:
                return llm_cat.strip()[:100]
        except Exception:
            pass
    return "Uncategorized"


REQUIRED_CSV_COLUMNS_CANDIDATES = {
    "date": ["date", "transaction date", "posted date", "trans date"],
    "amount": ["amount", "value", "debit", "amount (usd)", "transaction amount"],
    "vendor": ["vendor", "merchant", "description", "payee", "name"],
    "description": ["description", "details", "memo", "narration"],
}


def _find_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


def parse_csv_bytes(content: bytes) -> List[Dict[str, Any]]:
    """Parse a real bank/expense CSV export into normalized transaction dicts.
    Column names are auto-detected since real exports vary by bank/provider."""
    df = pd.read_csv(io.BytesIO(content))
    df.columns = [str(c).strip() for c in df.columns]

    date_col = _find_column(list(df.columns), REQUIRED_CSV_COLUMNS_CANDIDATES["date"])
    amount_col = _find_column(list(df.columns), REQUIRED_CSV_COLUMNS_CANDIDATES["amount"])
    vendor_col = _find_column(list(df.columns), REQUIRED_CSV_COLUMNS_CANDIDATES["vendor"])
    desc_col = _find_column(list(df.columns), REQUIRED_CSV_COLUMNS_CANDIDATES["description"])

    if amount_col is None:
        raise ValueError(
            f"Could not find an amount column in the CSV. Columns found: {list(df.columns)}"
        )

    records = []
    for _, row in df.iterrows():
        try:
            amount = float(str(row[amount_col]).replace(",", "").replace("$", ""))
        except (ValueError, TypeError):
            continue

        date_val = datetime.datetime.utcnow()
        if date_col is not None:
            try:
                date_val = pd.to_datetime(row[date_col]).to_pydatetime()
            except Exception:
                pass

        vendor = str(row[vendor_col]) if vendor_col else "Unknown"
        description = str(row[desc_col]) if desc_col else vendor

        records.append({
            "date": date_val,
            "amount": abs(amount),
            "vendor": vendor[:255],
            "description": description[:500],
        })
    return records
