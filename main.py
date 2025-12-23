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

app = FastAPI(title="MyTaxLead AI Worker", version="2.0.0")


def env(name: str, default: str = "") -> str:
    v = os.getenv(name, default)
    return v.strip() if isinstance(v, str) else default


AI_WORKER_TOKEN = env("AI_WORKER_TOKEN")
OPENAI_API_KEY = env("OPENAI_API_KEY")
OPENAI_MODEL = env("OPENAI_MODEL", "gpt-5.2")
TIMEOUT_SECS = int(env("HTTP_TIMEOUT", "180"))

# 0 => return all rows (careful: can get huge)
MAX_ROWS = int(env("MAX_ROWS", "0"))
MAX_SHEETS = int(env("MAX_SHEETS", "5"))

# If you want the LLM to help categorise, keep sample size reasonable
LLM_SAMPLE_ROWS = int(env("LLM_SAMPLE_ROWS", "250"))
LLM_MAX_CHARS = int(env("LLM_MAX_CHARS", "180000"))


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
    # Converts NaN/Infinity to None so JSON encoder doesn't crash
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
    Accepts: dd/mm/yyyy, yyyy-mm-dd, dd Mon yyyy, etc.
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

    return None


def infer_currency(text: str) -> str:
    t = (text or "").upper()
    if " GBP" in t or "£" in t:
        return "GBP"
    if " EUR" in t or "€" in t:
        return "EUR"
    if " USD" in t or "$" in t:
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
# PDF parsing into rows (heuristic)
# ----------------------------
def parse_pdf_bytes(b: bytes) -> Dict[str, Any]:
    from io import BytesIO
    from PyPDF2 import PdfReader

    reader = PdfReader(BytesIO(b))
    pages = []
    full_text_parts = []
    for i, p in enumerate(reader.pages[:20]):  # first 20 pages for statements
        txt = (p.extract_text() or "").strip()
        pages.append({"page": i + 1, "text": txt[:12000]})
        if txt:
            full_text_parts.append(txt)
    full_text = "\n".join(full_text_parts)
    out = {"pages": pages, "page_count": len(reader.pages), "full_text": full_text[:500000]}
    return _sanitize_jsonable(out)


def pdf_extract_transactions(full_text: str) -> List[Dict[str, Any]]:
    """
    Tries to parse statement tables like:
    Processed on | Created on | Type | Description | Paid out | Paid in | Balance
    Works best for ANNA/PayrNet style where lines contain dd Mon yyyy and amounts.
    """
    text = full_text or ""
    if not text:
        return []

    # normalise whitespace
    text = re.sub(r"\r", "\n", text)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

    # detect rows by lines containing a date and at least one amount and a balance
    # Example row fragments:
    # "17 Dec 2025 17 Dec 2025 FP YUMNAH AMANI AHMED AL MARWAEI INVESTM 500.00 11.79"
    date_pat = re.compile(r"\b(\d{1,2}\s+[A-Za-z]{3}\s+\d{4})\b")
    money_pat = re.compile(r"[-£]?\d[\d,]*\.\d{2}")

    rows: List[Dict[str, Any]] = []
    buffer = ""

    def flush(buf: str):
        buf = re.sub(r"\s+", " ", buf).strip()
        if not buf:
            return
        # find 2 dates (processed + created) if present
        dates = date_pat.findall(buf)
        processed = parse_date_any(dates[0]) if len(dates) >= 1 else None
        created = parse_date_any(dates[1]) if len(dates) >= 2 else None

        # find amounts (usually paid out, paid in, balance near end)
        amts = money_pat.findall(buf)
        if len(amts) < 2:
            return

        # Heuristic: last amount is balance, the two before might be paid in/out or amount+balance
        balance = to_float_money(amts[-1])
        paid_in = None
        paid_out = None
        amount = None

        # Try to identify if there are 3 amounts: out, in, balance
        if len(amts) >= 3:
            a1 = to_float_money(amts[-3])
            a2 = to_float_money(amts[-2])
            # Many statements show paid_out then paid_in (one blank often)
            # We can't see blanks after text extraction, so guess:
            # if one of them is 0.00 often means blank; keep both
            paid_out = a1
            paid_in = a2
            # derive amount: money in positive, money out negative
            if paid_in and paid_in != 0:
                amount = abs(paid_in)
            elif paid_out and paid_out != 0:
                amount = -abs(paid_out)
        else:
            # 2 amounts: amount and balance
            a = to_float_money(amts[-2])
            amount = a
            balance = balance

        # type code guess (FP / POS / CASH / FEE etc)
        type_code = ""
        m = re.search(r"\b(FP|POS|CASH|FEE|DD|SO|BACS|CHQ)\b", buf.upper())
        if m:
            type_code = m.group(1)

        # description: remove dates and trailing money bits
        desc = buf
        # strip leading dates
        desc = date_pat.sub("", desc).strip()
        # strip trailing money
        desc = re.sub(r"(?:\s+[-£]?\d[\d,]*\.\d{2}){2,}$", "", desc).strip()

        # also remove type code at start
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
        # start a new buffer when we see a line that looks like a row start (date)
        if date_pat.search(ln) and money_pat.search(ln):
            if buffer:
                flush(buffer)
            buffer = ln
        else:
            # continuation line
            if buffer:
                buffer += " " + ln

    if buffer:
        flush(buffer)

    return rows


# ----------------------------
# Normalisation + bookkeeping
# ----------------------------
def normalize_statement_rows(rows: List[Dict[str, Any]], source_kind: str) -> List[Dict[str, Any]]:
    """
    Standard output fields:
    - date, type, payee, description, money_in, money_out, amount, balance, reference
    """
    out: List[Dict[str, Any]] = []

    # column-mapping for CSV/XLSX is unknown; try to infer
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

            # fallback from amt sign
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

        raw_date = get("date","valuedate","transactiondate","postingdate","processedon","createdon")
        dt = parse_date_any(raw_date)

        desc = get("description","details","narrative","memo","merchant","name","transactiondetails")
        typ = get("type","transactiontype","code","typecode")
        ref = get("reference","ref","id","transactionid")

        bal = to_float_money(get("balance","runningbalance","availablebalance"))

        # prefer separate in/out columns
        raw_in = to_float_money(get("paidin","moneyin","credit","in"))
        raw_out = to_float_money(get("paidout","moneyout","debit","out"))

        amt = to_float_money(get("amount","value","transactionamount","net"))

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

    # sort by date then stable
    def sort_key(x):
        d = x.get("date") or "9999-12-31"
        return d
    out2.sort(key=sort_key)
    return _maybe_limit_rows(out2)


def compute_reconciliation(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    money_in = 0.0
    money_out = 0.0
    net = 0.0
    balances = [r.get("balance") for r in rows if isinstance(r.get("balance"), (int, float))]
    opening = balances[0] if balances else None
    closing = balances[-1] if balances else None

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
        recon_ok = abs(recon_diff) <= 0.02  # 2p tolerance

    return {
        "opening_balance_inferred": opening,
        "closing_balance_inferred": closing,
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
            dups.append({"row_index": i, "duplicate_of": seen[key], "key": list(key)})
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
        diff = (dates[i] - dates[i-1]).days
        if diff >= gap_days:
            gaps.append({"from": dates[i-1].isoformat(), "to": dates[i].isoformat(), "gap_days": diff})
    return gaps[:200]


def suspicious_flags(rows: List[Dict[str, Any]]) -> List[str]:
    flags = []
    if not rows:
        return ["no_transactions_detected"]

    # many cash
    cash_count = sum(1 for r in rows if (r.get("type") or "").upper() == "CASH" or "cash" in (r.get("description") or "").lower())
    if cash_count >= 10:
        flags.append(f"high_cash_activity({cash_count})")

    # many fees
    fee_count = sum(1 for r in rows if (r.get("type") or "").upper() == "FEE" or "fee" in (r.get("description") or "").lower() or "subscription" in (r.get("description") or "").lower())
    if fee_count >= 5:
        flags.append(f"recurring_fees({fee_count})")

    # unusually many small repetitive amounts
    small = [round(float(r["money_out"]), 2) for r in rows if isinstance(r.get("money_out"), (int, float)) and 0 < float(r["money_out"]) <= 5]
    if len(small) >= 15:
        flags.append("many_small_outgoing_payments(<=5)")

    # missing dates
    missing_dates = sum(1 for r in rows if not r.get("date"))
    if missing_dates > 0:
        flags.append(f"missing_dates({missing_dates})")

    return flags


def to_bookkeeping_csv(rows: List[Dict[str, Any]]) -> str:
    # Basic CSV for import
    import csv
    from io import StringIO

    cols = ["date","type","payee","description","money_in","money_out","balance","reference"]
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
# LLM layer: categorisation + SA/Company summaries
# ----------------------------
def llm_accountant_pack(
    meta: Dict[str, Any],
    normalized_rows: List[Dict[str, Any]],
    reconciliation: Dict[str, Any],
    currency_guess: str,
) -> Dict[str, Any]:
    # If no key, still return a deterministic structure
    if not OPENAI_API_KEY:
        return {
            "ok": True,
            "model": None,
            "summary": "OPENAI_API_KEY not set. Returning computed totals + flags only.",
            "doc_type": "unknown",
            "currency": currency_guess,
            "sa_summary": {},
            "company_accounts_summary": {},
            "categorised_sample": [],
            "issues": ["missing_openai_api_key"],
        }

    # Take sample for cost
    sample = normalized_rows[: max(1, LLM_SAMPLE_ROWS)]
    compact = {
        "meta": meta,
        "currency_guess": currency_guess,
        "reconciliation": reconciliation,
        "sample_transactions": sample,
        "notes": "Sample only (not full list). Use it to infer categories/rules.",
    }
    compact_json = json.dumps(compact, ensure_ascii=False)[:LLM_MAX_CHARS]

    prompt = """
You are an expert UK accountant and bookkeeping assistant.

You will receive:
- meta info
- reconciliation totals (money in/out/net, opening/closing inferred)
- a SAMPLE of transactions already normalised: date, type, description, money_in, money_out, balance

Return STRICT JSON ONLY with:
1) doc_type: bank_statement / card_statement / mixed / unknown
2) currency: GBP/EUR/USD/unknown
3) executive_summary: 5-10 bullet points (strings) suitable for an accountant
4) rules_inferred: list of rules you inferred for this statement format (columns meaning, how to treat type codes)
5) sa_summary: for Self Assessment (high-level):
   - total_income
   - total_allowable_expenses
   - net_profit
   - expense_breakdown: dict category->amount
6) company_accounts_summary:
   - turnover
   - cost_of_sales (if any)
   - operating_expenses (dict)
   - profit_before_tax
7) categorised_sample: list of objects matching the provided sample transactions, add:
   - category (e.g. "Sales", "Rent", "Bank fees", "Fuel", "Subscriptions", "Cash deposit", "Transfers", "Unknown")
   - confidence 0..1
   - notes (brief)
8) issues: list of strings for:
   - missing periods, duplicates, unclear currency, reconciliation mismatch, suspicious patterns, etc.
9) follow_up_questions: list of questions to ask the client if needed

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

        js = json.loads(text)
        if not isinstance(js, dict):
            raise ValueError("AI did not return a JSON object")

        js["ok"] = True
        js["model"] = OPENAI_MODEL
        return _sanitize_jsonable(js)

    except Exception as e:
        return {
            "ok": True,
            "model": OPENAI_MODEL,
            "summary": "AI call failed. Returning computed totals + flags only.",
            "error": str(e),
        }


def post_callback(callback_url: str, webhook_secret: str, payload: Dict[str, Any]) -> None:
    headers = {"Content-Type": "application/json", "X-AI-Webhook-Secret": webhook_secret}
    r = requests.post(callback_url, json=payload, headers=headers, timeout=TIMEOUT_SECS)
    r.raise_for_status()


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

    # ---------- Extract transactions rows (full)
    raw_rows: List[Dict[str, Any]] = []

    currency_guess = "unknown"

    if kind == "csv":
        raw_rows = list(extracted.get("data", {}).get("records", []) or [])
        # quick currency guess from values
        currency_guess = infer_currency(json.dumps(extracted.get("data", {})[:0] if False else extracted.get("data", {})))
    elif kind == "xlsx":
        sheets = extracted.get("data", {}).get("sheets", {}) or {}
        # take first sheet by default
        if sheets:
            first_sheet = list(sheets.keys())[0]
            raw_rows = list(sheets[first_sheet].get("records", []) or [])
        currency_guess = infer_currency(json.dumps(extracted.get("data", {})))
    elif kind == "pdf":
        full_text = extracted.get("data", {}).get("full_text", "") or ""
        currency_guess = infer_currency(full_text)
        raw_rows = pdf_extract_transactions(full_text)

    normalized = normalize_statement_rows(raw_rows, kind)

    # ---------- Reconciliation & checks
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

    # ---------- Bookkeeping CSV (string)
    bookkeeping_csv = to_bookkeeping_csv(normalized)

    # ---------- Accountant pack (LLM)
    meta = {
        "file": req.original_name,
        "kind": kind,
        "client_id": req.client_id,
        "upload_id": req.upload_id,
        "job_id": req.job_id,
    }
    accountant_pack = llm_accountant_pack(meta, normalized, recon, currency_guess)

    payload = {
        "job_id": req.job_id,
        "upload_id": req.upload_id,
        "client_id": req.client_id,
        "status": "done",
        "error": "",
        "extracted": extracted,  # still keep your raw extraction
        "normalized_transactions": normalized,
        "reconciliation": recon,
        "checks": checks,
        "bookkeeping_csv": bookkeeping_csv,  # PHP can save/export this
        "accountant_pack": accountant_pack,  # SA + company summary + categories (sample)
    }

    if req.callback_url and req.webhook_secret:
        try:
            post_callback(req.callback_url, req.webhook_secret, payload)
        except Exception as e:
            payload["warning"] = f"Callback failed: {e}"

    return _sanitize_jsonable(payload)