from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import Optional, Any, Dict
import os, requests, tempfile, mimetypes, json

import pandas as pd

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


app = FastAPI()

WORKER_TOKEN = os.environ.get("AI_WORKER_TOKEN", "")

class AnalyzeReq(BaseModel):
    stored_name: str
    client_id: Optional[int] = None
    file_url: str

@app.get("/health")
def health():
    return {"ok": True}

def auth_ok(auth_header: Optional[str]) -> bool:
    if not WORKER_TOKEN:
        return False
    if not auth_header:
        return False
    if not auth_header.lower().startswith("bearer "):
        return False
    token = auth_header.split(" ", 1)[1].strip()
    return token == WORKER_TOKEN

def read_pdf_text(path: str) -> str:
    if PdfReader is None:
        return ""
    try:
        reader = PdfReader(path)
        parts = []
        for p in reader.pages:
            parts.append(p.extract_text() or "")
        return "\n".join(parts).strip()
    except Exception:
        return ""

def read_csv_text(path: str) -> Dict[str, Any]:
    # try pandas auto-detect
    df = pd.read_csv(path)
    return {
        "rows": int(len(df)),
        "columns": list(df.columns)[:50],
        "head": df.head(20).to_dict(orient="records"),
    }

def read_excel_text(path: str) -> Dict[str, Any]:
    xl = pd.ExcelFile(path)
    sheets = xl.sheet_names[:10]
    preview = {}
    for s in sheets:
        df = xl.parse(s).head(20)
        preview[s] = df.to_dict(orient="records")
    return {"sheets": sheets, "preview": preview}

def basic_financial_guess(extracted: Dict[str, Any]) -> Dict[str, Any]:
    # MVP: we only return previews + basic flags.
    flags = []
    if extracted.get("file_type") == "pdf" and (extracted.get("text_len", 0) < 50):
        flags.append("PDF text looks empty (might be a scanned image). OCR not enabled yet.")
    return {"flags": flags}

@app.post("/analyze")
def analyze(req: AnalyzeReq, authorization: Optional[str] = Header(default=None)):
    if not auth_ok(authorization):
        raise HTTPException(status_code=401, detail="Unauthorized")

    # download file
    try:
        r = requests.get(req.file_url, timeout=60)
        if r.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Could not fetch file (HTTP {r.status_code})")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")

    # temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(r.content)
        tmp_path = tmp.name

    # guess type by content-type or name
    ctype = r.headers.get("content-type", "")
    guessed_ext = mimetypes.guess_extension(ctype.split(";")[0].strip()) or ""
    name = req.stored_name.lower()

    file_type = "unknown"
    if name.endswith(".pdf") or guessed_ext == ".pdf":
        file_type = "pdf"
    elif name.endswith(".csv") or guessed_ext == ".csv":
        file_type = "csv"
    elif name.endswith(".xlsx") or name.endswith(".xls") or guessed_ext in [".xlsx", ".xls"]:
        file_type = "excel"

    extracted: Dict[str, Any] = {"file_type": file_type}

    try:
        if file_type == "pdf":
            text = read_pdf_text(tmp_path)
            extracted["text_len"] = len(text)
            extracted["text_preview"] = text[:4000]
        elif file_type == "csv":
            extracted.update(read_csv_text(tmp_path))
        elif file_type == "excel":
            extracted.update(read_excel_text(tmp_path))
        else:
            extracted["bytes"] = len(r.content)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    insights = basic_financial_guess(extracted)

    # MVP response (we’ll evolve this to full bookkeeping later)
    summary = "AI parsed the file. Open the preview and review flags."
    if insights.get("flags"):
        summary = "AI parsed the file but found issues: " + "; ".join(insights["flags"])

    return {
        "status": "done",
        "client_id": req.client_id,
        "stored_name": req.stored_name,
        "file_type": file_type,
        "summary": summary,
        "insights": insights,
        "extracted": extracted
    }