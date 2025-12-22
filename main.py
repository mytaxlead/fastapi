import os
import json
import re
import math
import hashlib
from typing import Any, Dict, Optional, List, Tuple

import requests
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

app = FastAPI(title="MyTaxLead AI Worker", version="1.1.0")


def env(name: str, default: str = "") -> str:
    v = os.getenv(name, default)
    return v.strip() if isinstance(v, str) else default


AI_WORKER_TOKEN = env("AI_WORKER_TOKEN")
OPENAI_API_KEY = env("OPENAI_API_KEY")
OPENAI_MODEL = env("OPENAI_MODEL", "gpt-5.1")  # safe default
TIMEOUT_SECS = int(env("HTTP_TIMEOUT", "120"))


class AnalyzeRequest(BaseModel):
    job_id: int
    upload_id: int
    client_id: int
    original_name: str
    signed_url: str
    hint: Optional[str] = None


def require_token(authorization: Optional[str]) -> None:
    if not AI_WORKER_TOKEN:
        raise HTTPException(status_code=500, detail="AI_WORKER_TOKEN not set")
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if token != AI_WORKER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")


def download_to_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=TIMEOUT_SECS)
    r.raise_for_status()
    return r.content


def detect_kind(original_name: str, signed_url: str) -> str:
    name = (original_name or "").lower()
    if name.endswith(".pdf"):
        return "pdf"
    if name.endswith(".csv"):
        return "csv"
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return "xlsx"
    u = (signed_url or "").lower()
    for ext in (".pdf", ".csv", ".xlsx", ".xls"):
        if ext in u:
            return ext.replace(".", "")
    return "unknown"


# -----------------------------
# JSON safety (fix NaN/inf)
# -----------------------------
def _clean_json(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    if isinstance(x, (int, str, bool)):
        return x
    if isinstance(x, dict):
        return {str(k): _clean_json(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean_json(v) for v in x]
    return str(x)


# -----------------------------
# CSV parsing + accountant analysis
# -----------------------------
def _guess_columns(cols: List[str]) -> Dict[str, Optional[str]]:
    """
    Try to map unknown bank CSV headers to standard fields.
    Standard fields: date, description, amount, balance, type, reference
    """
    lc = {c: c.lower().strip() for c in cols}

    def pick(*needles) -> Optional[str]:
        for n in needles:
            for c, cl in lc.items():
                if cl == n or cl.endswith(n) or n in cl:
                    return c
        return None

    date_c = pick("date", "transaction date", "trans date", "posted date")
    desc_c = pick("description", "details", "narrative", "merchant", "name", "payee")
    amt_c = pick("amount", "amt", "value", "transaction amount")
    bal_c = pick("balance", "running balance", "closing balance")
    type_c = pick("type", "transaction type", "category")
    ref_c = pick("reference", "ref", "payment reference", "transaction reference")

    # Some exports have separate debit/credit columns
    debit_c = pick("debit", "money out", "paid out")
    credit_c = pick("credit", "money in", "paid in")

    return {
        "date": date_c,
        "description": desc_c,
        "amount": amt_c,
        "balance": bal_c,
        "type": type_c,
        "reference": ref_c,
        "debit": debit_c,
        "credit": credit_c,
    }


def _to_float_safe(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return float(v)
    s = str(v).strip()
    if s == "":
        return None
    # remove currency symbols/commas
    s = re.sub(r"[£$,]", "", s)
    # handle parentheses negatives e.g. (12.34)
    if re.match(r"^\(.*\)$", s):
        s = "-" + s[1:-1]
    try:
        return float(s)
    except Exception:
        return None


def _to_date_str(v: Any) -> Optional[str]:
    # Keep it simple: return original string trimmed. (Better parsing can be added later)
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _merchant_key(desc: str) -> str:
    d = (desc or "").lower()
    d = re.sub(r"[^a-z0-9\s]", " ", d)
    d = re.sub(r"\s+", " ", d).strip()
    # remove obvious noise tokens
    for t in ["payment", "paid", "transfer", "faster", "fps", "pos", "card", "contactless", "dd", "so"]:
        d = re.sub(rf"\b{t}\b", "", d).strip()
    d = re.sub(r"\s+", " ", d).strip()
    # keep first 3 words
    parts = d.split(" ")
    return " ".join(parts[:3]) if parts else ""


def parse_csv_bytes(b: bytes) -> Dict[str, Any]:
    import pandas as pd
    from io import BytesIO

    df = pd.read_csv(BytesIO(b))

    cols = list(df.columns.astype(str))
    m = _guess_columns(cols)

    # Build normalized transactions
    txns: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        date_v = row.get(m["date"]) if m["date"] else None
        desc_v = row.get(m["description"]) if m["description"] else None
        type_v = row.get(m["type"]) if m["type"] else None
        ref_v = row.get(m["reference"]) if m["reference"] else None

        # Amount logic: prefer single amount column; else credit/debit pair
        amount = None
        if m["amount"]:
            amount = _to_float_safe(row.get(m["amount"]))
        else:
            credit = _to_float_safe(row.get(m["credit"])) if m["credit"] else None
            debit = _to_float_safe(row.get(m["debit"])) if m["debit"] else None
            # common pattern: debit positive means money out
            if credit is not None and debit is None:
                amount = credit
            elif debit is not None and credit is None:
                amount = -abs(debit)
            elif credit is not None and debit is not None:
                # if both present, net them
                amount = (credit or 0.0) - (debit or 0.0)

        balance = _to_float_safe(row.get(m["balance"])) if m["balance"] else None

        txn = {
            "date": _to_date_str(date_v),
            "description": (str(desc_v).strip() if desc_v is not None else None),
            "type": (str(type_v).strip() if type_v is not None else None),
            "reference": (str(ref_v).strip() if ref_v is not None else None),
            "amount": amount,
            "balance": balance,
        }

        # Skip empty rows
        if not any([txn["date"], txn["description"], txn["amount"], txn["balance"]]):
            continue

        txns.append(txn)

    # Basic stats
    amounts = [t["amount"] for t in txns if isinstance(t.get("amount"), (int, float)) and t["amount"] is not None]
    credits = [a for a in amounts if a > 0]
    debits = [a for a in amounts if a < 0]

    total_in = float(sum(credits)) if credits else 0.0
    total_out = float(sum(abs(x) for x in debits)) if debits else 0.0
    net = float(total_in - total_out)

    # Date range (string-based; later can parse properly)
    dates = [t["date"] for t in txns if t.get("date")]
    date_start = min(dates) if dates else None
    date_end = max(dates) if dates else None

    # Largest items
    largest_in = sorted(
        [t for t in txns if (t.get("amount") or 0) > 0],
        key=lambda x: x.get("amount") or 0,
        reverse=True
    )[:10]
    largest_out = sorted(
        [t for t in txns if (t.get("amount") or 0) < 0],
        key=lambda x: x.get("amount") or 0
    )[:10]

    # Detect recurring merchants (simple heuristic)
    merchant_counts: Dict[str, int] = {}
    merchant_examples: Dict[str, str] = {}
    for t in txns:
        desc = t.get("description") or ""
        mk = _merchant_key(desc)
        if not mk:
            continue
        merchant_counts[mk] = merchant_counts.get(mk, 0) + 1
        merchant_examples.setdefault(mk, desc)

    recurring = [
        {"merchant_key": k, "count": c, "example": merchant_examples.get(k)}
        for k, c in sorted(merchant_counts.items(), key=lambda kv: kv[1], reverse=True)
        if c >= 3
    ][:20]

    # Simple flags
    fee_like = []
    cash_like = []
    subscription_like = []
    for t in txns:
        d = (t.get("description") or "").lower()
        if any(w in d for w in ["fee", "charge", "commission", "interest"]):
            fee_like.append(t)
        if any(w in d for w in ["cash", "atm"]):
            cash_like.append(t)
        if any(w in d for w in ["subscription", "subs", "netflix", "spotify", "prime", "apple", "google", "microsoft"]):
            subscription_like.append(t)

    analysis = {
        "date_range": {"start": date_start, "end": date_end},
        "counts": {
            "transactions": len(txns),
            "credits": len(credits),
            "debits": len(debits),
        },
        "totals": {
            "total_in": total_in,
            "total_out": total_out,
            "net": net,
        },
        "largest": {
            "largest_in": largest_in,
            "largest_out": largest_out,
        },
        "recurring_merchants": recurring,
        "flags": {
            "possible_fees": fee_like[:30],
            "cash_related": cash_like[:30],
            "possible_subscriptions": subscription_like[:30],
        }
    }

    # Return full transactions too (admin needs it)
    out = {
        "columns_detected": m,
        "transactions": txns,         # full list
        "analysis": analysis,         # accountant-friendly summary stats
        "preview": txns[:50],         # quick preview
    }

    return _clean_json(out)


def parse_xlsx_bytes(b: bytes) -> Dict[str, Any]:
    import pandas as pd
    from io import BytesIO

    xls = pd.ExcelFile(BytesIO(b))
    sheets = {}
    for s in xls.sheet_names[:5]:
        df = xls.parse(s)
        sheets[s] = {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "columns": list(df.columns.astype(str)),
            "preview": df.head(30).to_dict(orient="records"),
        }
    return _clean_json({"sheets": sheets, "sheet_names": xls.sheet_names})


def parse_pdf_bytes(b: bytes) -> Dict[str, Any]:
    from io import BytesIO
    from PyPDF2 import PdfReader

    reader = PdfReader(BytesIO(b))
    pages = []
    for i, p in enumerate(reader.pages[:10]):
        txt = (p.extract_text() or "").strip()
        pages.append({"page": i + 1, "text": txt[:6000]})
    return _clean_json({"pages": pages, "page_count": len(reader.pages)})


def llm_summary(extracted: Dict[str, Any], original_name: str) -> Dict[str, Any]:
    # If no OpenAI key, still return structured "non-AI" summary.
    if not OPENAI_API_KEY:
        return {
            "ok": True,
            "model": None,
            "summary": "OPENAI_API_KEY not set. Showing extracted analysis only.",
            "doc_type": "unknown",
            "key_fields": {},
            "issues": ["missing_openai_api_key"],
        }

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        prompt = f"""
You are an accountant assistant.
Given extracted data from a file named: {original_name}

Return STRICT JSON only with these keys:
- summary: short human summary for an accountant
- doc_type: e.g. bank_statement_csv, bank_statement_pdf, invoice, payslip, unknown
- key_fields: dict of important extracted fields (names, dates, totals, account hints)
- issues: list of possible issues/missing info
- suggested_checks: list of checks accountant should do
"""

        resp = client.responses.create(
            model=OPENAI_MODEL or "gpt-5.1",
            input=[
                {"role": "system", "content": "Return STRICT JSON only. No markdown."},
                {"role": "user", "content": prompt},
                {"role": "user", "content": json.dumps(extracted)[:150000]},
            ],
        )

        text = ""
        if hasattr(resp, "output_text"):
            text = resp.output_text
        else:
            for item in getattr(resp, "output", []) or []:
                if item.get("type") == "message":
                    for c in item.get("content", []):
                        if c.get("type") == "output_text":
                            text += c.get("text", "")

        text = (text or "").strip()
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            text = m.group(0)

        try:
            js = json.loads(text)
            js["ok"] = True
            js["model"] = OPENAI_MODEL
            return _clean_json(js)
        except Exception:
            return {
                "ok": True,
                "model": OPENAI_MODEL,
                "summary": "AI returned non-JSON. Showing raw output.",
                "raw": text,
            }

    except Exception as e:
        return {
            "ok": True,
            "model": OPENAI_MODEL,
            "summary": "AI call failed. Showing extracted analysis only.",
            "error": str(e),
        }


@app.get("/")
def root():
    return {"ok": True, "service": "mytaxlead-ai-worker"}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/debug_token")
def debug_token():
    t = AI_WORKER_TOKEN or ""
    return {
        "len": len(t),
        "sha256": hashlib.sha256(t.encode("utf-8")).hexdigest(),
        "starts": t[:2],
        "ends": t[-2:],
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest, authorization: Optional[str] = Header(default=None)):
    require_token(authorization)

    kind = detect_kind(req.original_name, req.signed_url)
    b = download_to_bytes(req.signed_url)

    extracted: Dict[str, Any] = {
        "kind": kind,
        "original_name": req.original_name,
        "upload_id": req.upload_id,
        "client_id": req.client_id,
        "job_id": req.job_id,
    }

    try:
        if kind == "csv":
            extracted["data"] = parse_csv_bytes(b)
        elif kind == "xlsx":
            extracted["data"] = parse_xlsx_bytes(b)
        elif kind == "pdf":
            extracted["data"] = parse_pdf_bytes(b)
        else:
            extracted["data"] = {"note": "Unknown file type", "bytes": len(b)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Parse failed: {e}")

    summary = llm_summary(extracted, req.original_name)

    return _clean_json({
        "ok": True,
        "job_id": req.job_id,
        "upload_id": req.upload_id,
        "client_id": req.client_id,
        "extracted": extracted,
        "summary": summary,
    })