import os
import json
import time
import requests
import tempfile
from typing import Any, Dict, Optional

import pandas as pd
from pypdf import PdfReader
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

app = FastAPI()

WORKER_TOKEN = os.environ.get("AI_WORKER_TOKEN", "")
WEBHOOK_SECRET = os.environ.get("AI_WEBHOOK_SECRET", "")

class JobIn(BaseModel):
    job_id: int
    upload_id: int
    client_id: int
    original_name: str
    stored_name: str
    file_mime: str
    file_size: int
    download_url: str
    callback_url: str

def _auth_ok(auth_header: Optional[str]) -> bool:
    if not WORKER_TOKEN:
        return False
    if not auth_header:
        return False
    if not auth_header.lower().startswith("bearer "):
        return False
    token = auth_header.split(" ", 1)[1].strip()
    return token == WORKER_TOKEN

def send_callback(callback_url: str, payload: Dict[str, Any]) -> None:
    headers = {"Content-Type": "application/json"}
    if WEBHOOK_SECRET:
        headers["X-AI-Webhook-Secret"] = WEBHOOK_SECRET
    requests.post(callback_url, headers=headers, data=json.dumps(payload), timeout=30)

def download_file(url: str) -> str:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    fd, path = tempfile.mkstemp(prefix="mtl_", suffix=".bin")
    with os.fdopen(fd, "wb") as f:
        f.write(r.content)
    return path

def extract_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    out = []
    for page in reader.pages[:30]:  # cap pages
        try:
            out.append(page.extract_text() or "")
        except Exception:
            out.append("")
    return "\n".join(out).strip()

def extract_csv(path: str) -> pd.DataFrame:
    # try common encodings
    for enc in ["utf-8", "latin-1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path, engine="python")

def extract_xlsx(path: str) -> pd.DataFrame:
    # first sheet
    return pd.read_excel(path)

def naive_bank_statement_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Very simple "day one" extraction:
    - tries to find debit/credit/amount columns
    - returns totals + row count + date range if possible
    """
    cols = [c.lower().strip() for c in df.columns]
    def find_col(names):
        for n in names:
            for i, c in enumerate(cols):
                if n in c:
                    return df.columns[i]
        return None

    debit_col  = find_col(["debit", "money out", "paid out", "out"])
    credit_col = find_col(["credit", "money in", "paid in", "in"])
    amount_col = find_col(["amount", "amt", "value"])

    total_debit = 0.0
    total_credit = 0.0

    def to_num(x):
        try:
            if pd.isna(x): return 0.0
            s = str(x).replace(",", "").replace("£","").strip()
            if s == "": return 0.0
            return float(s)
        except Exception:
            return 0.0

    if debit_col:
        total_debit = float(df[debit_col].map(to_num).sum())
    if credit_col:
        total_credit = float(df[credit_col].map(to_num).sum())

    # If only Amount exists, infer +/- if there's a "type" or sign
    if not (debit_col or credit_col) and amount_col:
        vals = df[amount_col].map(to_num)
        total_credit = float(vals[vals > 0].sum())
        total_debit  = float((-vals[vals < 0]).sum())

    # date range attempt
    date_col = find_col(["date", "transaction date", "posted"])
    date_min = date_max = None
    if date_col:
        try:
            d = pd.to_datetime(df[date_col], errors="coerce")
            date_min = str(d.min().date()) if pd.notna(d.min()) else None
            date_max = str(d.max().date()) if pd.notna(d.max()) else None
        except Exception:
            pass

    return {
        "rows": int(len(df)),
        "date_col": date_col,
        "date_min": date_min,
        "date_max": date_max,
        "total_money_in": round(total_credit, 2),
        "total_money_out": round(total_debit, 2),
    }

@app.get("/")
def root():
    return {"ok": True, "service": "mytaxlead-ai-worker"}

@app.post("/jobs")
def create_job(job: JobIn, authorization: Optional[str] = Header(default=None)):
    if not _auth_ok(authorization):
        raise HTTPException(status_code=401, detail="Unauthorized")

    # tell PHP we started
    try:
        send_callback(job.callback_url, {
            "job_id": job.job_id,
            "upload_id": job.upload_id,
            "client_id": job.client_id,
            "status": "processing",
        })
    except Exception:
        pass

    try:
        path = download_file(job.download_url)

        extracted: Dict[str, Any] = {
            "file": {
                "original_name": job.original_name,
                "stored_name": job.stored_name,
                "mime": job.file_mime,
                "size": job.file_size,
            }
        }
        summary: Dict[str, Any] = {
            "review_notes": [],
            "figures": {},
        }

        ext = (job.original_name or "").lower().split(".")[-1]

        if ext == "pdf" or "pdf" in (job.file_mime or ""):
            text = extract_pdf_text(path)
            extracted["pdf_text_preview"] = text[:20000]
            summary["review_notes"].append("PDF extracted as text preview (basic). For scanned PDFs we will add OCR later.")
            summary["figures"]["detected_type"] = "pdf"

        elif ext in ["csv"]:
            df = extract_csv(path)
            extracted["columns"] = list(df.columns)
            extracted["head"] = df.head(20).to_dict(orient="records")
            summary["figures"]["detected_type"] = "csv"
            summary["figures"]["bank_statement"] = naive_bank_statement_summary(df)

        elif ext in ["xlsx", "xls"]:
            df = extract_xlsx(path)
            extracted["columns"] = list(df.columns)
            extracted["head"] = df.head(20).to_dict(orient="records")
            summary["figures"]["detected_type"] = "excel"
            summary["figures"]["bank_statement"] = naive_bank_statement_summary(df)

        else:
            # fallback: just store bytes size info
            summary["review_notes"].append(f"Unsupported file type for day-one parser: {ext}")
            summary["figures"]["detected_type"] = "unknown"

        # send done
        send_callback(job.callback_url, {
            "job_id": job.job_id,
            "upload_id": job.upload_id,
            "client_id": job.client_id,
            "status": "done",
            "extracted": extracted,
            "summary": summary
        })
        return {"ok": True}

    except Exception as e:
        try:
            send_callback(job.callback_url, {
                "job_id": job.job_id,
                "upload_id": job.upload_id,
                "client_id": job.client_id,
                "status": "error",
                "error_text": str(e)[:2000],
            })
        except Exception:
            pass
        raise HTTPException(status_code=500, detail="Job failed")