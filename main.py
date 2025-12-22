import os
import json
import re
import hashlib
from typing import Any, Dict, Optional

import requests
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

app = FastAPI(title="MyTaxLead AI Worker", version="1.0.0")


def env(name: str, default: str = "") -> str:
    v = os.getenv(name, default)
    return v.strip() if isinstance(v, str) else default


AI_WORKER_TOKEN = env("AI_WORKER_TOKEN")
OPENAI_API_KEY = env("OPENAI_API_KEY")
OPENAI_MODEL = env("OPENAI_MODEL", "gpt-5.2")  # good default
TIMEOUT_SECS = int(env("HTTP_TIMEOUT", "120"))


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
    # fallback: try from url
    u = (signed_url or "").lower()
    for ext in (".pdf", ".csv", ".xlsx", ".xls"):
        if ext in u:
            return ext.replace(".", "")
    return "unknown"


def parse_csv_bytes(b: bytes) -> Dict[str, Any]:
    import pandas as pd
    from io import BytesIO

    df = pd.read_csv(BytesIO(b))
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": list(df.columns.astype(str)),
        "preview": df.head(50).to_dict(orient="records"),
    }


def parse_xlsx_bytes(b: bytes) -> Dict[str, Any]:
    import pandas as pd
    from io import BytesIO

    xls = pd.ExcelFile(BytesIO(b))
    sheets = {}
    for s in xls.sheet_names[:5]:
        df = xls.parse(s).head(50)
        sheets[s] = {
            "rows_preview": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "columns": list(df.columns.astype(str)),
            "preview": df.to_dict(orient="records"),
        }
    return {"sheets": sheets, "sheet_names": xls.sheet_names}


def parse_pdf_bytes(b: bytes) -> Dict[str, Any]:
    from io import BytesIO
    from PyPDF2 import PdfReader

    reader = PdfReader(BytesIO(b))
    pages = []
    for i, p in enumerate(reader.pages[:8]):
        txt = (p.extract_text() or "").strip()
        pages.append({"page": i + 1, "text": txt[:6000]})
    return {"pages": pages, "page_count": len(reader.pages)}


def llm_summary(extracted: Dict[str, Any], original_name: str) -> Dict[str, Any]:
    # If no OpenAI key, return basic summary so pipeline still works.
    if not OPENAI_API_KEY:
        return {
            "ok": True,
            "model": None,
            "summary": "OPENAI_API_KEY not set. Returning extracted preview only.",
            "doc_type": "unknown",
            "key_fields": {},
            "issues": ["missing_openai_api_key"],
        }

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        prompt = f"""
You are an accounting document assistant.

Given extracted data from a file named: {original_name}

Return STRICT JSON ONLY with these keys:
- summary: short human summary
- doc_type: e.g. bank_statement, sales_report, invoice_list, payroll, unknown
- key_fields: dict of important fields found (account name, period, totals, balances, etc.)
- issues: list of issues/missing info
- transactions_hint: if this contains transactions, describe the columns and what they mean
"""

        resp = client.responses.create(
            model=OPENAI_MODEL or "gpt-5.2",
            input=[
                {"role": "system", "content": "Return STRICT JSON only. No markdown."},
                {"role": "user", "content": prompt},
                {"role": "user", "content": json.dumps(extracted)[:180000]},
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
            if not isinstance(js, dict):
                raise ValueError("Not a JSON object")
            js["ok"] = True
            js["model"] = OPENAI_MODEL
            return js
        except Exception:
            return {
                "ok": True,
                "model": OPENAI_MODEL,
                "summary": "AI returned non-JSON. Returning raw output.",
                "raw": text,
            }

    except Exception as e:
        return {
            "ok": True,
            "model": OPENAI_MODEL,
            "summary": "AI call failed. Returning extracted preview only.",
            "error": str(e),
        }


def post_callback(callback_url: str, webhook_secret: str, payload: Dict[str, Any]) -> None:
    headers = {
        "Content-Type": "application/json",
        "X-AI-Webhook-Secret": webhook_secret,
    }
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

    # Build extracted payload
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
        # callback error too
        if req.callback_url and req.webhook_secret:
            try:
                post_callback(req.callback_url, req.webhook_secret, {
                    "job_id": req.job_id,
                    "upload_id": req.upload_id,
                    "client_id": req.client_id,
                    "status": "error",
                    "error": f"Parse failed: {e}",
                })
            except Exception:
                pass
        raise HTTPException(status_code=400, detail=f"Parse failed: {e}")

    summary = llm_summary(extracted, req.original_name)

    # Always callback if provided
    if req.callback_url and req.webhook_secret:
        status = "done"
        err_txt = ""
        if isinstance(summary, dict) and summary.get("error"):
            # still "done" is ok, but you can flip to error if you want
            err_txt = str(summary.get("error"))

        try:
            post_callback(req.callback_url, req.webhook_secret, {
                "job_id": req.job_id,
                "upload_id": req.upload_id,
                "client_id": req.client_id,
                "status": status,
                "error": err_txt,
                "extracted": extracted,
                "summary": summary,
            })
        except Exception as e:
            # If callback fails, we still return ok to caller, but include warning
            return {
                "ok": True,
                "warning": f"Callback failed: {e}",
                "job_id": req.job_id,
            }

    return {
        "ok": True,
        "job_id": req.job_id,
        "upload_id": req.upload_id,
        "client_id": req.client_id,
    }