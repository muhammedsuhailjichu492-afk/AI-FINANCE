
import os
import re
import datetime
from typing import Optional, Dict, Any

import pytesseract
from PIL import Image

try:
    from pdf2image import convert_from_path
    PDF_SUPPORT = True
except Exception:
    PDF_SUPPORT = False

AMOUNT_PATTERNS = [
    r"(?:total|amount due|grand total|balance due|amount)\s*[:\-]?\s*\$?\s*([\d,]+\.\d{2})",
    r"\$\s*([\d,]+\.\d{2})",
    r"([\d,]+\.\d{2})\s*(?:usd|inr|eur|gbp)",
]

DATE_PATTERNS = [
    r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
    r"(\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})",
    r"((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})",
]

VENDOR_STOPWORDS = {"invoice", "receipt", "bill", "statement", "tax", "total", "date"}


def extract_text(filepath: str) -> str:
    """Run OCR on an image or PDF file and return raw extracted text."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        if not PDF_SUPPORT:
            raise RuntimeError(
                "PDF support requires poppler to be installed and on PATH. "
                "See README for Windows install instructions."
            )
        pages = convert_from_path(filepath, dpi=300)
        text = "\n".join(pytesseract.image_to_string(p) for p in pages)
    else:
        img = Image.open(filepath)
        text = pytesseract.image_to_string(img)
    return text


def _parse_amount(text: str) -> Optional[float]:
    lower = text.lower()
    best = None
    for pattern in AMOUNT_PATTERNS:
        matches = re.findall(pattern, lower)
        for m in matches:
            try:
                val = float(m.replace(",", ""))
                if best is None or val > best:
                    best = val
            except ValueError:
                continue
        if best is not None:
            break
    return best


def _parse_date(text: str) -> datetime.datetime:
    for pattern in DATE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            raw = match.group(1)
            for fmt in ("%m/%d/%Y", "%m-%d-%Y", "%d/%m/%Y", "%Y-%m-%d", "%Y/%m/%d",
                        "%m/%d/%y", "%b %d, %Y", "%B %d, %Y", "%b. %d, %Y"):
                try:
                    return datetime.datetime.strptime(raw.strip(), fmt)
                except ValueError:
                    continue
    return datetime.datetime.utcnow()


def _parse_vendor(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in lines[:6]:
        cleaned = re.sub(r"[^A-Za-z0-9&.,'\s]", "", line).strip()
        if len(cleaned) < 3:
            continue
        if cleaned.lower() in VENDOR_STOPWORDS:
            continue
        if re.search(r"\d{4,}", cleaned):  # looks like a phone/id number line
            continue
        return cleaned[:120]
    return "Unknown Vendor"


def parse_document(filepath: str) -> Dict[str, Any]:
    """Full document-processing step: OCR + field extraction."""
    raw_text = extract_text(filepath)
    amount = _parse_amount(raw_text)
    date = _parse_date(raw_text)
    vendor = _parse_vendor(raw_text)
    return {
        "raw_text": raw_text,
        "amount": amount,
        "date": date,
        "vendor": vendor,
        "description": " ".join(raw_text.split())[:500],
    }
