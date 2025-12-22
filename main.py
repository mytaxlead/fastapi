import os, time, json, re
import requests
import pandas as pd
import pdfplumber
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# OpenAI (official SDK)
from openai import OpenAI

app = FastAPI()

def getenv(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()

AI_WORKER_TOKEN = getenv("AI_WORKER_TOKEN")  # you already set this in Railway Variables
OPENAI_API_KEY  = getenv("OPENAI_API_KEY")
OPENAI_MODEL    = getenv("OPENAI_MODEL", "gpt-5.2")  # you can use gpt-5.2 (see note below)

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

class AnalyzeRequest(BaseModel):
    job_id: int | None = None
    upload_id: int | None = None
    client_id: int | None = None

    signed_url: str
    original_name: str | None = None
    mime: str | None = None

def require_token(auth_header: str | None):
    if not AI_WORKER_TOKEN:
        # Allow boot without token only if you forgot to set it (but you SHOULD set it)
        raise HTTPException(status_code=500, detail="AI_WORKER_TOKEN missing on server")
    if not auth_header or not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = auth_header.split(" ", 1)[1].strip()
    if token != AI_WORKER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

@app.get("/health")
def health():
    return {"ok": True, "ts": int(time.time())}

def detect_kind(filename: str, mime: str | None) -> str:
    fn = (filename or "").lower()
    if fn.endswith(".csv"):
        return "csv"
    if fn.endswith(".xlsx") or fn.endswith(".xls"):
        return "xlsx"
    if fn.endswith(".pdf"):
        return "pdf"
    # fallback by mime
    if mime:
        m = mime.lower()
        if "csv" in m:
            return "csv"
        if "excel" in m or "spreadsheet" in m:
            return "xlsx"
        if "pdf" in m:
            return "pdf"
    return "unknown"

def download_to_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content

def parse_csv_bytes(b: bytes) -> dict:
    # try common separators
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(pd.io.common.BytesIO(b), sep=sep)
            if df.shape[1] >= 2:
                return summarize_dataframe(df)
        except Exception:
            pass
    raise ValueError("Could not parse CSV")

def parse_xlsx_bytes(b: bytes) -> dict:
    df = pd.read_excel(pd.io.common.BytesIO(b))
    return summarize_dataframe(df)

def summarize_dataframe(df: pd.DataFrame) -> dict:
    # Make columns easier
    cols = [str(c).strip() for c in df.columns]
    df.columns = cols

    # Try to find an amount column
    amount_col = None
    for c in cols:
        lc = c.lower()
        if lc in ["amount", "amt", "value", "transaction amount", "debit/credit"]:
            amount_col = c
            break

    # Backup: find numeric-ish column with most numeric values
    if amount_col is None:
        best = (None, 0)
        for c in cols:
            s = pd.to_numeric(df[c], errors="coerce")
            score = s.notna().sum()
            if score > best[1]:
                best = (c, score)
        amount_col = best[0]

    out = {
        "rows": int(len(df)),
        "columns": cols[:50],
        "amount_column": amount_col,
    }

    if amount_col:
        s = pd.to_numeric(df[amount_col], errors="coerce").dropna()
        if len(s) > 0:
            out["amount_sum"] = float(s.sum())
            out["amount_min"] = float(s.min())
            out["amount_max"] = float(s.max())

    # Return a small preview to help the LLM (not full statement)
    preview = df.head(20).fillna("").astype(str).to_dict(orient="records")
    out["preview_rows"] = preview
    return out

def parse_pdf_bytes(b: bytes) -> dict:
    text_chunks = []
    with pdfplumber.open(pd.io.common.BytesIO(b)) as pdf:
        for i, page in enumerate(pdf.pages[:12]):  # limit pages
            try:
                t = page.extract_text() or ""
                t = re.sub(r"\s+", " ", t).strip()
                if t:
                    text_chunks.append({"page": i+1, "text": t[:4000]})
            except Exception:
                pass
    return {
        "pages_scanned": min(12, len(text_chunks) if text_chunks else 0),
        "text_sample": text_chunks[:8],
    }

def llm_summary(extracted: dict, filename: str) -> dict:
    if client is None:
        return {
            "note": "OPENAI_API_KEY not set on Railway, so only extraction ran.",
            "suggested_checks": ["Set OPENAI_API_KEY to enable AI summaries."]
        }

    prompt = f"""
You are an accounting assistant. The user uploaded a file: {filename}.
You are given extracted structured data (preview + basic stats). Your job:

1) Identify likely statement type: bank statement / sales report / expenses / payroll / other.
2) Pull out the key figures an accountant wants:
   - total inflows, total outflows (if possible)
   - net movement
   - any VAT-like totals if hinted
   - date range if hinted
3) Flag anything suspicious or needing manual review (missing columns, weird amounts, etc.)
4) Output STRICT JSON.

Extracted:
{json.dumps(extracted)[:120000]}
"""

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
    )

    # responses API returns text; we enforce “json-ish” by asking strict JSON
    txt = resp.output_text.strip()

    # best effort: if it returned text around json, try to extract a json object
    m = re.search(r"\{.*\}\s*$", txt, re.S)
    if m:
        txt = m.group(0)

    try:
        return json.loads(txt)
    except Exception:
        return {"raw": txt}

@app.post("/analyze")
def analyze(req: AnalyzeRequest, authorization: str | None = Header(default=None)):
    require_token(authorization)

    kind = detect_kind(req.original_name or "", req.mime)
    b = download_to_bytes(req.signed_url)

    extracted = {"kind": kind, "original_name": req.original_name, "mime": req.mime}

    try:
        if kind == "csv":
            extracted["data"] = parse_csv_bytes(b)
        elif kind == "xlsx":
            extracted["data"] = parse_xlsx_bytes(b)
        elif kind == "pdf":
            extracted["data"] = parse_pdf_bytes(b)
        else:
            extracted["data"] = {"note": "Unknown file type. Add support if needed."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Parse failed: {str(e)}")

    summary = llm_summary(extracted, req.original_name or "uploaded_file")

    return {
        "ok": True,
        "job_id": req.job_id,
        "upload_id": req.upload_id,
        "client_id": req.client_id,
        "extracted": extracted,
        "summary": summary,
    }