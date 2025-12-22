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
OPENAI_MODEL = env("OPENAI_MODEL", "gpt-5.1")
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


def parse_csv_bytes(b: bytes) -> Dict[str, Any]:
    import pandas as pd
    from io import BytesIO

    df = pd.read_csv(BytesIO(b))
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": list(df.columns.astype(str)),
        "preview": df.head(20).to_dict(orient="records"),
    }


def parse_xlsx_bytes(b: bytes) -> Dict[str, Any]:
    import pandas as pd
    from io import BytesIO

    xls = pd.ExcelFile(BytesIO(b))
    sheets = {}
    for s in xls.sheet_names[:5]:
        df = xls.parse(s).head(20)
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
    for i, p in enumerate(reader.pages[:5]):
        txt = (p.extract_text() or "").strip()
        pages.append({"page": i + 1, "text": txt[:4000]})
    return {"pages": pages, "page_count": len(reader.pages)}


def llm_summary(extracted: Dict[str, Any], original_name: str) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        return {
            "ok": True,
            "model": None,
            "summary": "OPENAI_API_KEY not set. Returning non-AI extracted preview only.",
            "actions": [],
            "flags": ["missing_openai_api_key"],
        }

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        prompt = f"""
You are an accounting document assistant.
Given extracted data from a file named: {original_name}
Return JSON only with:
- summary: short human summary
- doc_type: what kind of document it is
- key_fields: dict of important fields found
- issues: list of possible issues/missing info
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
            return {"ok": True, "model": OPENAI_MODEL, **js}
        except Exception:
            return {"ok": True, "model": OPENAI_MODEL, "summary": "AI returned non-JSON.", "raw": text}

    except Exception as e:
        return {"ok": True, "model": OPENAI_MODEL, "summary": "AI call failed.", "error": str(e)}


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

    return {
        "ok": True,
        "job_id": req.job_id,
        "upload_id": req.upload_id,
        "client_id": req.client_id,
        "extracted": extracted,
        "summary": summary,
    }