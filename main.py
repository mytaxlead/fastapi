import os
import json
import re
import math
import hashlib
from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime, date

import requests
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

app = FastAPI(title="MyTaxLead AI Worker", version="2.1.0")


# ----------------------------
# Env + config
# ----------------------------
def env(name: str, default: str = "") -> str:
    v = os.getenv(name, default)
    return v.strip() if isinstance(v, str) else default


AI_WORKER_TOKEN = env("AI_WORKER_TOKEN")
OPENAI_API_KEY = env("OPENAI_API_KEY")
OPENAI_MODEL = env("OPENAI_MODEL", "gpt-5.2")
TIMEOUT_SECS = int(env("HTTP_TIMEOUT", "180"))

# 0 => return all rows (careful: huge payloads)
MAX_ROWS = int(env("MAX_ROWS", "0"))
MAX_SHEETS = int(env("MAX_SHEETS", "5"))

# LLM sampling / safety
LLM_SAMPLE_ROWS = int(env("LLM_SAMPLE_ROWS", "250"))
LLM_MAX_CHARS = int(env("LLM_MAX_CHARS", "180000"))

# Optional: disable debug endpoints in prod
ENABLE_DEBUG = env("ENABLE_DEBUG", "0") in ("1", "true", "yes")


class AnalyzeRequest(BaseModel):
    job_id: int
    upload_id: int
    client_id: int
    original_name: str
    stored_name: Optional[str] = None
    signed_url: str

    callback_url: Optional[str] = None
    webhook_secret: Optional[str] = None
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


def _sanitize_jsonable(x: Any) -> Any:
    """Converts NaN/Infinity to None so JSON encoder doesn't crash."""
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    if isinstance(x, dict):
        return {str(k): _sanitize_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_sanitize_jsonable(v) for v in x]
    return x


def _maybe_limit_rows(rows: List[dict]) -> List[dict]:
    if MAX_ROWS and MAX_ROWS > 0:
        return rows[:MAX_ROWS]
    return rows


# ----------------------------
# Money + date parsing helpers
# ----------------------------
_money_strip_re = re.compile(r"[^\d\-\.\,]")


def to_float_money(v: Any) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    s = _money_strip_re.sub("", s)

    # 1,234.56 vs 1.234,56
    if "," in s and "." in s:
        # assume commas are thousand separators
        s = s.replace(",", "")
    elif "," in s and "." not in s:
        # comma decimal
        s = s.replace(",", ".")
    if not re.fullmatch(r"-?\d+(\.\d+)?", s or ""):
        return None
    try:
        return float(s)
    except Exception:
        return None


def parse_date_any(v: Any) -> Optional[str]:
    """
    Return ISO date string YYYY-MM-DD when possible.
    Accepts: dd/mm/yyyy, yyyy-mm-dd, dd Mon yyyy, and a few datetime forms.
    """
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None

    # dd/mm/yyyy
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", s)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return date(y, mo, d).isoformat()
        except Exception:
            return None

    # yyyy-mm-dd
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return date(y, mo, d).isoformat()
        except Exception:
            return None

    # dd Mon yyyy
    m = re.match(r"^(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})$", s)
    if m:
        d = int(m.group(1))
        mon = m.group(2).lower()[:3]
        y = int(m.group(3))
        months = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
        mo = months.get(mon)
        if mo:
            try:
                return date(y, mo, d).isoformat()
            except Exception:
                return None

    # common datetime-ish
    for fmt in ("%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except Exception:
            pass

    return None


def infer_currency(text: str) -> str:
    t = (text or "").upper()
    if "GBP" in t or "£" in t:
        return "GBP"
    if "EUR" in t or "€" in t:
        return "EUR"
    if "USD" in t or "$" in t:
        return "USD"
    return "unknown"


# ----------------------------
# CSV/XLSX parsing (full rows)
# ----------------------------
def parse_csv_bytes(b: bytes) -> Dict[str, Any]:
    import pandas as pd
    from io import BytesIO

    df = pd.read_csv(BytesIO(b), dtype=str, keep_default_na=False)
    records = df.to_dict(orient="records")
    records = _maybe_limit_rows(records)
    out = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": list(df.columns.astype(str)),
        "records": records,
    }
    return _sanitize_jsonable(out)


def parse_xlsx_bytes(b: bytes) -> Dict[str, Any]:
    import pandas as pd
    from io import BytesIO

    xls = pd.ExcelFile(BytesIO(b))
    sheet_names = xls.sheet_names
    sheets_out: Dict[str, Any] = {}

    for s in sheet_names[: max(1, MAX_SHEETS)]:
        df = xls.parse(s, dtype=str, keep_default_na=False)
        records = df.to_dict(orient="records")
        records = _maybe_limit_rows(records)
        sheets_out[s] = {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "columns": list(df.columns.astype(str)),
            "records": records,
        }

    out = {"sheet_names": sheet_names, "sheets": sheets_out}
    return _sanitize_jsonable(out)


# ----------------------------
# PDF parsing
# ----------------------------
def parse_pdf_bytes(b: bytes) -> Dict[str, Any]:
    from io import BytesIO
    from PyPDF2 import PdfReader

    reader = PdfReader(BytesIO(b))
    pages = []
    full_text_parts = []
    for i, p in enumerate(reader.pages[:20]):  # first 20 pages for typical statements
        txt = (p.extract_text() or "").strip()
        pages.append({"page": i + 1, "text": txt[:12000]})
        if txt:
            full_text_parts.append(txt)
    full_text = "\n".join(full_text_parts)
    out = {"pages": pages, "page_count": len(reader.pages), "full_text": full_text[:500000]}
    return _sanitize_jsonable(out)


def pdf_extract_transactions(full_text: str) -> List[Dict[str, Any]]:
    """
    Heuristic parser for statement-like PDFs.
    Builds rows with:
      processed_date, created_date, type, description, paid_out, paid_in, amount, balance
    """
    text = full_text or ""
    if not text:
        return []

    text = re.sub(r"\r", "\n", text)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

    date_pat = re.compile(r"\b(\d{1,2}\s+[A-Za-z]{3}\s+\d{4})\b")
    money_pat = re.compile(r"[-£]?\d[\d,]*\.\d{2}")

    rows: List[Dict[str, Any]] = []
    buffer = ""

    def flush(buf: str):
        buf = re.sub(r"\s+", " ", buf).strip()
        if not buf:
            return

        dates = date_pat.findall(buf)
        processed = parse_date_any(dates[0]) if len(dates) >= 1 else None
        created = parse_date_any(dates[1]) if len(dates) >= 2 else None

        amts = money_pat.findall(buf)
        if len(amts) < 2:
            return

        balance = to_float_money(amts[-1])
        paid_in = None
        paid_out = None
        amount = None

        if len(amts) >= 3:
            a1 = to_float_money(amts[-3])
            a2 = to_float_money(amts[-2])
            paid_out = a1
            paid_in = a2
            if paid_in and paid_in != 0:
                amount = abs(paid_in)
            elif paid_out and paid_out != 0:
                amount = -abs(paid_out)
        else:
            a = to_float_money(amts[-2])
            amount = a

        type_code = ""
        m = re.search(r"\b(FP|POS|CASH|FEE|DD|SO|BACS|CHQ)\b", buf.upper())
        if m:
            type_code = m.group(1)

        desc = buf
        desc = date_pat.sub("", desc).strip()
        desc = re.sub(r"(?:\s+[-£]?\d[\d,]*\.\d{2}){2,}$", "", desc).strip()
        if type_code:
            desc = re.sub(rf"^\b{re.escape(type_code)}\b\s*", "", desc, flags=re.I).strip()

        rows.append({
            "processed_date": processed,
            "created_date": created or processed,
            "type": type_code or None,
            "description": desc[:400],
            "paid_out": paid_out,
            "paid_in": paid_in,
            "amount": amount,
            "balance": balance,
        })

    for ln in lines:
        if date_pat.search(ln) and money_pat.search(ln):
            if buffer:
                flush(buffer)
            buffer = ln
        else:
            if buffer:
                buffer += " " + ln

    if buffer:
        flush(buffer)

    return rows


# ----------------------------
# Normalisation
# ----------------------------
def normalize_statement_rows(rows: List[Dict[str, Any]], source_kind: str) -> List[Dict[str, Any]]:
    """
    Standard output fields:
      date, type, payee, description, money_in, money_out, amount, balance, reference
    """
    out: List[Dict[str, Any]] = []

    def norm_key(k: str) -> str:
        k = (k or "").strip().lower()
        k = re.sub(r"[^a-z0-9]+", "", k)
        return k

    for r in rows:
        if not isinstance(r, dict):
            continue

        # If already shaped (PDF parser)
        if "processed_date" in r or "created_date" in r:
            dt = r.get("created_date") or r.get("processed_date")
            typ = r.get("type")
            desc = r.get("description") or ""
            pin = r.get("paid_in")
            pout = r.get("paid_out")
            bal = r.get("balance")
            amt = r.get("amount")

            mi = abs(pin) if isinstance(pin, (int, float)) and pin else None
            mo = abs(pout) if isinstance(pout, (int, float)) and pout else None

            if mi is None and mo is None and isinstance(amt, (int, float)):
                if amt >= 0:
                    mi = amt
                else:
                    mo = abs(amt)

            out.append({
                "date": dt,
                "type": typ,
                "payee": None,
                "description": desc,
                "money_in": mi,
                "money_out": mo,
                "amount": (mi if mi is not None else (-mo if mo is not None else amt)),
                "balance": bal,
                "reference": None,
            })
            continue

        # Otherwise infer from generic columns
        keys = {norm_key(k): k for k in r.keys()}

        def get(*cands):
            for c in cands:
                if c in keys:
                    return r.get(keys[c])
            return None

        raw_date = get("date", "valuedate", "transactiondate", "postingdate", "processedon", "createdon")
        dt = parse_date_any(raw_date)

        desc = get("description", "details", "narrative", "memo", "merchant", "name", "transactiondetails")
        typ = get("type", "transactiontype", "code", "typecode")
        ref = get("reference", "ref", "id", "transactionid")

        bal = to_float_money(get("balance", "runningbalance", "availablebalance"))

        raw_in = to_float_money(get("paidin", "moneyin", "credit", "in", "paidingbp", "ingbp"))
        raw_out = to_float_money(get("paidout", "moneyout", "debit", "out", "paidoutgbp", "outgbp"))

        amt = to_float_money(get("amount", "value", "transactionamount", "net", "amountsigned"))

        mi = abs(raw_in) if raw_in is not None and raw_in != 0 else None
        mo = abs(raw_out) if raw_out is not None and raw_out != 0 else None

        if mi is None and mo is None and amt is not None:
            if amt >= 0:
                mi = amt
            else:
                mo = abs(amt)

        out.append({
            "date": dt,
            "type": (str(typ).strip() if typ is not None and str(typ).strip() else None),
            "payee": None,
            "description": (str(desc).strip() if desc is not None else ""),
            "money_in": mi,
            "money_out": mo,
            "amount": (mi if mi is not None else (-mo if mo is not None else amt)),
            "balance": bal,
            "reference": (str(ref).strip() if ref is not None and str(ref).strip() else None),
        })

    # drop empty rows
    out2 = []
    for r in out:
        if r.get("date") or r.get("description") or (r.get("money_in") is not None) or (r.get("money_out") is not None):
            out2.append(r)

    # sort by date (unknown dates go last)
    def sort_key(x):
        return x.get("date") or "9999-12-31"

    out2.sort(key=sort_key)
    return _maybe_limit_rows(out2)


# ----------------------------
# Reconciliation + checks
# ----------------------------
def compute_period(rows: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    ds = [r.get("date") for r in rows if isinstance(r.get("date"), str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", r["date"])]
    if not ds:
        return {"start": None, "end": None}
    return {"start": min(ds), "end": max(ds)}


def compute_reconciliation(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    money_in = 0.0
    money_out = 0.0
    net = 0.0

    opening = None
    closing = None
    for r in rows:
        if isinstance(r.get("balance"), (int, float)) and (r.get("date") is not None):
            opening = float(r["balance"])
            break
    for r in reversed(rows):
        if isinstance(r.get("balance"), (int, float)) and (r.get("date") is not None):
            closing = float(r["balance"])
            break

    for r in rows:
        mi = r.get("money_in")
        mo = r.get("money_out")
        if isinstance(mi, (int, float)):
            money_in += float(mi)
            net += float(mi)
        if isinstance(mo, (int, float)):
            money_out += float(mo)
            net -= float(mo)

    recon_ok = None
    recon_diff = None
    if opening is not None and closing is not None:
        recon_diff = (opening + net) - closing
        recon_ok = abs(recon_diff) <= 0.02

    return {
        "opening_balance_inferred": (round(opening, 2) if opening is not None else None),
        "closing_balance_inferred": (round(closing, 2) if closing is not None else None),
        "total_money_in": round(money_in, 2),
        "total_money_out": round(money_out, 2),
        "net_movement": round(net, 2),
        "reconciles": recon_ok,
        "reconcile_diff": (round(recon_diff, 2) if recon_diff is not None else None),
    }


def find_duplicates(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = {}
    dups = []
    for i, r in enumerate(rows):
        key = (
            r.get("date"),
            (r.get("description") or "").strip().lower()[:120],
            round(float(r["money_in"]), 2) if isinstance(r.get("money_in"), (int, float)) else 0.0,
            round(float(r["money_out"]), 2) if isinstance(r.get("money_out"), (int, float)) else 0.0,
            round(float(r["balance"]), 2) if isinstance(r.get("balance"), (int, float)) else None,
        )
        if key in seen:
            dups.append({"row_index": i, "duplicate_of": seen[key]})
        else:
            seen[key] = i
    return dups[:500]


def find_missing_periods(rows: List[Dict[str, Any]], gap_days: int = 30) -> List[Dict[str, Any]]:
    dates = []
    for r in rows:
        ds = r.get("date")
        if ds:
            try:
                dates.append(datetime.strptime(ds, "%Y-%m-%d").date())
            except Exception:
                pass
    dates.sort()
    gaps = []
    for i in range(1, len(dates)):
        diff = (dates[i] - dates[i - 1]).days
        if diff >= gap_days:
            gaps.append({"from": dates[i - 1].isoformat(), "to": dates[i].isoformat(), "gap_days": diff})
    return gaps[:200]


def suspicious_flags(rows: List[Dict[str, Any]]) -> List[str]:
    flags = []
    if not rows:
        return ["no_transactions_detected"]

    cash_count = sum(
        1 for r in rows
        if (r.get("type") or "").upper() == "CASH" or "cash" in (r.get("description") or "").lower()
    )
    if cash_count >= 10:
        flags.append(f"high_cash_activity({cash_count})")

    fee_count = sum(
        1 for r in rows
        if (r.get("type") or "").upper() == "FEE"
        or "fee" in (r.get("description") or "").lower()
        or "charge" in (r.get("description") or "").lower()
        or "subscription" in (r.get("description") or "").lower()
    )
    if fee_count >= 5:
        flags.append(f"recurring_fees({fee_count})")

    small = [
        round(float(r["money_out"]), 2)
        for r in rows
        if isinstance(r.get("money_out"), (int, float)) and 0 < float(r["money_out"]) <= 5
    ]
    if len(small) >= 15:
        flags.append("many_small_outgoing_payments(<=5)")

    missing_dates = sum(1 for r in rows if not r.get("date"))
    if missing_dates > 0:
        flags.append(f"missing_dates({missing_dates})")

    return flags


# ----------------------------
# Bookkeeping CSV
# ----------------------------
def to_bookkeeping_csv(rows: List[Dict[str, Any]]) -> str:
    import csv
    from io import StringIO

    cols = ["date", "type", "payee", "description", "money_in", "money_out", "balance", "reference"]
    buf = StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    for r in rows:
        w.writerow([
            r.get("date") or "",
            r.get("type") or "",
            r.get("payee") or "",
            (r.get("description") or "").replace("\n", " ").strip(),
            (f"{r.get('money_in'):.2f}" if isinstance(r.get("money_in"), (int, float)) else ""),
            (f"{r.get('money_out'):.2f}" if isinstance(r.get("money_out"), (int, float)) else ""),
            (f"{r.get('balance'):.2f}" if isinstance(r.get("balance"), (int, float)) else ""),
            r.get("reference") or "",
        ])
    return buf.getvalue()


# ----------------------------
# LLM layer (optional enhancement)
# ----------------------------
def llm_accountant_pack(
    meta: Dict[str, Any],
    normalized_rows: List[Dict[str, Any]],
    reconciliation: Dict[str, Any],
    currency_guess: str,
) -> Dict[str, Any]:
    base = {
        "ok": True,
        "model": None,
        "doc_type": "unknown",
        "currency": currency_guess,
        "executive_summary": [],
        "rules_inferred": [],
        "sa_summary": {},
        "company_accounts_summary": {},
        "categorised_sample": [],
        "issues": [],
        "follow_up_questions": [],
    }

    if not OPENAI_API_KEY:
        base["summary"] = "OPENAI_API_KEY not set. Returning computed totals + flags only."
        base["issues"] = ["missing_openai_api_key"]
        return base

    sample = normalized_rows[: max(1, LLM_SAMPLE_ROWS)]
    compact = {
        "meta": meta,
        "currency_guess": currency_guess,
        "reconciliation": reconciliation,
        "sample_transactions": sample,
        "notes": "Sample only. Use it to infer categories/rules and questions.",
    }
    compact_json = json.dumps(compact, ensure_ascii=False)[:LLM_MAX_CHARS]

    prompt = """
You are an expert UK accountant and bookkeeping assistant.

You will receive:
- meta info
- reconciliation totals (money in/out/net, opening/closing inferred)
- a SAMPLE of transactions already normalised: date, type, description, money_in, money_out, balance

Return STRICT JSON ONLY with keys:
doc_type, currency,
executive_summary (list of strings),
rules_inferred (list of strings),
sa_summary (object: total_income, total_allowable_expenses, net_profit, expense_breakdown dict),
company_accounts_summary (object: turnover, cost_of_sales, operating_expenses dict, profit_before_tax),
categorised_sample (list of objects for each sample row: category, confidence 0..1, notes),
issues (list of strings),
follow_up_questions (list of strings).

Be conservative: if unsure, category="Unknown" and add an issue.
No markdown. JSON object only.
"""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        resp = client.responses.create(
            model=OPENAI_MODEL or "gpt-5.2",
            input=[
                {"role": "system", "content": "Return STRICT JSON only. No markdown."},
                {"role": "user", "content": prompt.strip()},
                {"role": "user", "content": compact_json},
            ],
        )

        text = getattr(resp, "output_text", "") or ""
        text = text.strip()
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            text = m.group(0)

        js = json.loads(text)
        if not isinstance(js, dict):
            raise ValueError("AI did not return a JSON object")

        js["ok"] = True
        js["model"] = OPENAI_MODEL
        return _sanitize_jsonable(js)

    except Exception as e:
        base["model"] = OPENAI_MODEL
        base["summary"] = "AI call failed. Returning computed totals + flags only."
        base["error"] = str(e)
        base["issues"] = ["llm_failed"]
        return base


def post_callback(callback_url: str, webhook_secret: str, payload: Dict[str, Any]) -> None:
    headers = {"Content-Type": "application/json", "X-AI-Webhook-Secret": webhook_secret}
    r = requests.post(callback_url, json=payload, headers=headers, timeout=TIMEOUT_SECS)
    r.raise_for_status()


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def root():
    return {"ok": True, "service": "mytaxlead-ai-worker"}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/debug_token")
def debug_token():
    if not ENABLE_DEBUG:
        raise HTTPException(status_code=404, detail="Not found")
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
        "stored_name": req.stored_name,
        "upload_id": req.upload_id,
        "client_id": req.client_id,
        "job_id": req.job_id,
    }

    # ---------- Parse
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
        payload = {
            "job_id": req.job_id,
            "upload_id": req.upload_id,
            "client_id": req.client_id,
            "status": "error",
            "error": f"Parse failed: {e}",
        }
        if req.callback_url and req.webhook_secret:
            try:
                post_callback(req.callback_url, req.webhook_secret, payload)
            except Exception:
                pass
        raise HTTPException(status_code=400, detail=payload["error"])

    # ---------- Extract raw rows
    raw_rows: List[Dict[str, Any]] = []
    currency_guess = "unknown"

    if kind == "csv":
        raw_rows = list(extracted.get("data", {}).get("records", []) or [])
        currency_guess = infer_currency(json.dumps(extracted.get("data", {}) or {}, ensure_ascii=False))
    elif kind == "xlsx":
        sheets = extracted.get("data", {}).get("sheets", {}) or {}
        if sheets:
            first_sheet = list(sheets.keys())[0]
            raw_rows = list(sheets[first_sheet].get("records", []) or [])
        currency_guess = infer_currency(json.dumps(extracted.get("data", {}) or {}, ensure_ascii=False))
    elif kind == "pdf":
        full_text = extracted.get("data", {}).get("full_text", "") or ""
        currency_guess = infer_currency(full_text)
        raw_rows = pdf_extract_transactions(full_text)

    normalized = normalize_statement_rows(raw_rows, kind)

    # ---------- Totals & checks
    period = compute_period(normalized)
    recon = compute_reconciliation(normalized)
    dups = find_duplicates(normalized)
    gaps = find_missing_periods(normalized, gap_days=30)
    sus = suspicious_flags(normalized)

    checks = {
        "duplicates": dups,
        "missing_period_gaps": gaps,
        "suspicious_flags": sus,
        "row_count_normalized": len(normalized),
    }

    # ---------- Bookkeeping CSV
    bookkeeping_csv = to_bookkeeping_csv(normalized)

    # ---------- Accountant pack (LLM enhancement)
    meta = {
        "file": req.original_name,
        "kind": kind,
        "client_id": req.client_id,
        "upload_id": req.upload_id,
        "job_id": req.job_id,
        "period": period,
    }
    accountant_pack = llm_accountant_pack(meta, normalized, recon, currency_guess)

    payload = {
        "job_id": req.job_id,
        "upload_id": req.upload_id,
        "client_id": req.client_id,
        "status": "done",
        "error": "",

        # keep compatible outputs for your PHP
        "extracted": extracted,
        "normalized_transactions": normalized,
        "reconciliation": recon,
        "checks": checks,
        "bookkeeping_csv": bookkeeping_csv,

        # optional LLM pack
        "accountant_pack": accountant_pack,
    }

    if req.callback_url and req.webhook_secret:
        try:
            post_callback(req.callback_url, req.webhook_secret, payload)
        except Exception as e:
            payload["warning"] = f"Callback failed: {e}"

    return _sanitize_jsonable(payload)