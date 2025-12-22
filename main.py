import os
import json
import re
from typing import Optional, Any, Dict

import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# ----------------------------
# Env
# ----------------------------
WORKER_TOKEN = os.getenv("AI_WORKER_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "").strip() or "gpt-5.2"

app = FastAPI(title="MyTaxLead AI Worker", version="1.0.0")


class AnalyzeRequest(BaseModel):
    job_id: int
    upload_id: int
    client_id: int
    original_name: str
    signed_url: str  # your PHP signed download URL


def require_token(authorization: Optional[str]):
    if not WORKER_TOKEN:
        # If you forgot to set the token in Railway Variables
        raise HTTPException(status_code=500, detail="AI_WORKER_TOKEN not set")

    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    m = re.match(r"Bearer\s+(.+)", authorization.strip(), re.I)
    if not m:
        raise HTTPException(status_code=401, detail="Invalid Authorization header")

    if m.group(1).strip() != WORKER_TOKEN:
        raise HTTPException(status_code=403, detail="Bad token")


def detect_kind(name: str) -> str:
    n = (name or "").lower()
    if n.endswith(".csv"):
        return "csv"
    if n.endswith(".xlsx") or n.endswith(".xls"):
        return "xlsx"
    if n.endswith(".pdf"):
        return "pdf"
    return "unknown"


async def download_to_bytes(url: str) -> bytes:
    timeout = httpx.Timeout(60.0, connect=15.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        r = await client.get(url)
        if r.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Download failed: HTTP {r.status_code}")
        return r.content


def parse_csv_bytes(b: bytes) -> Dict[str, Any]:
    # Keep lightweight: return a preview only (first ~200 lines)
    text = b.decode("utf-8", errors="replace")
    lines = text.splitlines()
    preview = lines[:200]
    return {"lines_preview": preview, "line_count": len(lines)}


def parse_xlsx_bytes(_: bytes) -> Dict[str, Any]:
    # For day-one simplicity, we don't parse XLSX locally.
    # We send raw bytes info + filename to the model for instructions / summary.
    return {"note": "XLSX received. (Local parsing can be added later.)"}


def parse_pdf_bytes(_: bytes) -> Dict[str, Any]:
    # For day-one simplicity, we don't OCR/PDF parse locally.
    return {"note": "PDF received. (Text extraction/OCR can be added later.)"}


async def llm_summary(extracted: Dict[str, Any], original_name: str) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")

    # We ask for bookkeeping-relevant fields in JSON.
    # This is a first version; we’ll refine schema later.
    system = (
        "You are an accounting assistant. "
        "Given extracted data from a customer document (bank statement / CSV / PDF notice / spreadsheet), "
        "return JSON with: "
        "document_type, period_start, period_end, currency, totals (credits, debits, closing_balance if possible), "
        "flags (missing_pages, unreadable, password_protected, inconsistent_dates), "
        "and a short 'admin_review_notes'. "
        "If data is insufficient, set fields to null and explain in admin_review_notes."
    )

    user_payload = {
        "original_name": original_name,
        "extracted": extracted,
    }

    # Using OpenAI Responses API via the Python SDK.
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload)},
        ],
        # Keep deterministic-ish for bookkeeping extraction
        temperature=0.2,
    )

    text = resp.output_text or ""
    # Try to find a JSON object in the response
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {"raw": text.strip()}

    try:
        return json.loads(m.group(0))
    except Exception:
        return {"raw": text.strip()}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/analyze")
async def analyze(req: AnalyzeRequest, authorization: Optional[str] = Header(None)):
    require_token(authorization)

    kind = detect_kind(req.original_name)
    b = await download_to_bytes(req.signed_url)

    extracted: Dict[str, Any] = {
        "kind": kind,
        "original_name": req.original_name,
        "bytes_len": len(b),
    }

    if kind == "csv":
        extracted["data"] = parse_csv_bytes(b)
    elif kind == "xlsx":
        extracted["data"] = parse_xlsx_bytes(b)
    elif kind == "pdf":
        extracted["data"] = parse_pdf_bytes(b)
    else:
        extracted["data"] = {"note": "Unknown file type"}

    summary = await llm_summary(extracted, req.original_name)

    return {
        "ok": True,
        "job_id": req.job_id,
        "upload_id": req.upload_id,
        "client_id": req.client_id,
        "extracted": extracted,
        "summary": summary,
    }