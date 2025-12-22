import os
import json
import re
import math
import hashlib
from typing import Any, Dict, Optional, List

import requests
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

app = FastAPI(title="MyTaxLead AI Worker", version="1.1.0")


def env(name: str, default: str = "") -> str:
    v = os.getenv(name, default)
    return v.strip() if isinstance(v, str) else default


AI_WORKER_TOKEN = env("AI_WORKER_TOKEN")
OPENAI_API_KEY = env("OPENAI_API_KEY")
# gpt-5.2 is valid per OpenAI model docs; leave blank in Railway to use default here.
OPENAI_MODEL = env("OPENAI_MODEL", "gpt-5.2")
TIMEOUT_SECS = int(env("HTTP_TIMEOUT", "180"))

# If 0 => ALL rows (what you asked for).
# If you ever hit memory issues, set MAX_ROWS=5000 etc in Railway variables.
MAX_ROWS = int(env("MAX_ROWS", "0"))

# For XLSX, you may have multiple sheets. Default "all" is heavy.
# Set MAX_SHEETS=1 for only first sheet.
MAX_SHEETS = int(env("MAX_SHEETS", "5"))


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


def parse_csv_bytes(b: bytes) -> Dict[str, Any]:
    import pandas as pd
    from io import BytesIO

    # Keep as strings where possible (bank statements can have weird formatting)
    df = pd.read_csv(BytesIO(b), dtype=str, keep_default_na=False)
    records = df.to_dict(orient="records")
    records = _maybe_limit_rows(records)

    out = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": list(df.columns.astype(str)),
        "records": records,  # <-- FULL rows here (or limited if MAX_ROWS set)
    }
    return _sanitize_jsonable(out)


def parse_xlsx_bytes(b: bytes) -> Dict[str, Any]:
    import pandas as pd
    from io import BytesIO

    xls = pd.ExcelFile(BytesIO(b))
    sheet_names = xls.sheet_names
    sheets_out: Dict[str, Any] = {}

    # Process first N sheets (default 5)
    for s in sheet_names[: max(1, MAX_SHEETS)]:
        df = xls.parse(s, dtype=str, keep_default_na=False)
        records = df.to_dict(orient="records")
        records = _maybe_limit_rows(records)

        sheets_out[s] = {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "columns": list(df.columns.astype(str)),
            "records": records,  # <-- FULL rows here
        }

    out = {"sheet_names": sheet_names, "sheets": sheets_out}
    return _sanitize_jsonable(out)


def parse_pdf_bytes(b: bytes) -> Dict[str, Any]:
    # PDFs are not “rows”; we extract text.
    from io import BytesIO
    from PyPDF2 import PdfReader

    reader = PdfReader(BytesIO(b))
    pages = []
    for i, p in enumerate(reader.pages[:10]):
        txt = (p.extract_text() or "").strip()
        pages.append({"page": i + 1, "text": txt[:12000]})
    out = {"pages": pages, "page_count": len(reader.pages)}
    return _sanitize_jsonable(out)


def llm_summary(extracted: Dict[str, Any], original_name: str) -> Dict[str, Any]:
    # If no OpenAI key, pipeline still works and admin can view extracted.
    if not OPENAI_API_KEY:
        return {
            "ok": True,
            "model": None,
            "summary": "OPENAI_API_KEY not set. Showing extracted data only.",
            "doc_type": "unknown",
            "key_fields": {},
            "issues": ["missing_openai_api_key"],
        }

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        prompt = f"""
You are an accounting document assistant.

File name: {original_name}

Return STRICT JSON ONLY:
- summary
- doc_type (bank_statement / card_statement / invoice / payroll / unknown)
- key_fields (account name, bank, period, opening balance, closing balance, totals)
- issues (missing pages, unclear currency, duplicates, etc.)
- transactions_hint (what the columns likely mean)
"""

        # IMPORTANT: don't send *all* rows to the model (too expensive/slow).
        # We send a compact sample + column names and totals.
        compact = extracted.copy()
        try:
            if compact.get("kind") == "csv" and "data" in compact:
                data = compact["data"]
                cols = data.get("columns", [])
                recs = data.get("records", [])
                compact["data"] = {"columns": cols, "sample": recs[:200], "rows": data.get("rows"), "cols": data.get("cols")}
            if compact.get("kind") == "xlsx" and "data" in compact:
                data = compact["data"]
                sheets = data.get("sheets", {})
                new_sheets = {}
                for sn, sd in list(sheets.items())[:3]:
                    new_sheets[sn] = {"columns": sd.get("columns", []), "sample": (sd.get("records", [])[:200]), "rows": sd.get("rows"), "cols": sd.get("cols")}
                compact["data"] = {"sheet_names": data.get("sheet_names", []), "sheets": new_sheets}
        except Exception:
            pass

        resp = client.responses.create(
            model=OPENAI_MODEL or "gpt-5.2",
            input=[
                {"role": "system", "content": "Return STRICT JSON only. No markdown."},
                {"role": "user", "content": prompt},
                {"role": "user", "content": json.dumps(compact)[:180000]},
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
            "summary": "AI call failed. Showing extracted data only.",
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

    summary = llm_summary(extracted, req.original_name)

    payload = {
        "job_id": req.job_id,
        "upload_id": req.upload_id,
        "client_id": req.client_id,
        "status": "done",
        "error": "",
        "extracted": extracted,
        "summary": summary,
    }

    # Callback back to your PHP webhook (preferred)
    if req.callback_url and req.webhook_secret:
        try:
            post_callback(req.callback_url, req.webhook_secret, payload)
        except Exception as e:
            payload["warning"] = f"Callback failed: {e}"

    return payload