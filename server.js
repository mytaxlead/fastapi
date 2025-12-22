import express from "express";
import OpenAI from "openai";
import Papa from "papaparse";
import XLSX from "xlsx";

const app = express();
app.use(express.json({ limit: "5mb" }));

const PORT = process.env.PORT || 3000;

const WORKER_TOKEN = process.env.AI_WORKER_TOKEN || "";
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || "";
const MODEL = process.env.OPENAI_MODEL || "gpt-5";

if (!WORKER_TOKEN) console.warn("Missing AI_WORKER_TOKEN (Railway env).");
if (!OPENAI_API_KEY) console.warn("Missing OPENAI_API_KEY (Railway env).");

const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

function requireBearer(req, res) {
  const auth = req.headers.authorization || "";
  const ok = auth.startsWith("Bearer ") && auth.slice(7) === WORKER_TOKEN;
  if (!ok) {
    res.status(403).json({ error: "Forbidden" });
    return false;
  }
  return true;
}

function extOf(name) {
  const m = String(name || "").toLowerCase().match(/\.([a-z0-9]+)$/);
  return m ? m[1] : "";
}

async function fetchAsBuffer(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Fetch failed ${r.status}`);
  const ab = await r.arrayBuffer();
  return Buffer.from(ab);
}

async function analyzeWithOpenAI({ fileUrl, fileName, textHint }) {
  // PDFs: send as input_file (file_url) per OpenAI docs 
  const isPdf = extOf(fileName) === "pdf";

  const content = [];
  content.push({
    type: "input_text",
    text:
      "You are an accounting assistant. Extract key bookkeeping figures and anomalies.\n" +
      "Return STRICT JSON with:\n" +
      "{\n" +
      '  "doc_type": "...",\n' +
      '  "period": {"from":"YYYY-MM-DD|null","to":"YYYY-MM-DD|null"},\n' +
      '  "currency":"GBP|...|unknown",\n' +
      '  "totals": {"credits":number|null,"debits":number|null,"net":number|null},\n' +
      '  "balances": {"opening":number|null,"closing":number|null},\n' +
      '  "vat": {"vatable_sales":number|null,"vatable_purchases":number|null,"vat_due":number|null},\n' +
      '  "flags": [{"level":"info|warn|high","message":"..."}],\n' +
      '  "suggested_actions": ["..."],\n' +
      '  "confidence": 0-1\n' +
      "}\n" +
      (textHint ? `\n\nExtra extracted text/table:\n${textHint}` : "")
  });

  if (isPdf) {
    content.push({ type: "input_file", file_url: fileUrl });
  }

  const resp = await openai.responses.create({
    model: MODEL,
    input: [{ role: "user", content }],
    // Ask for JSON-only output
    response_format: { type: "json_object" }
  });

  const out = resp.output_text || "{}";
  return JSON.parse(out);
}

app.post("/analyze", async (req, res) => {
  if (!requireBearer(req, res)) return;

  const { job_id, upload_id, client_id, original_name, stored_name, file_url, webhook_url } = req.body || {};
  if (!job_id || !upload_id || !client_id || !file_url || !webhook_url) {
    return res.status(400).json({ error: "Missing fields" });
  }

  // Reply immediately; continue async
  res.json({ ok: true, accepted: true });

  // Run async
  try {
    const name = original_name || stored_name || "file";
    const ext = extOf(name);

    let textHint = "";

    // For CSV/XLSX: download and convert to compact text summary
    if (ext === "csv" || ext === "xlsx" || ext === "xls") {
      const buf = await fetchAsBuffer(file_url);

      if (ext === "csv") {
        const csv = buf.toString("utf8");
        const parsed = Papa.parse(csv, { header: true, skipEmptyLines: true });
        const rows = (parsed.data || []).slice(0, 200); // cap
        textHint = "CSV rows (first 200):\n" + JSON.stringify(rows);
      } else {
        const wb = XLSX.read(buf, { type: "buffer" });
        const sheetName = wb.SheetNames[0];
        const ws = wb.Sheets[sheetName];
        const json = XLSX.utils.sheet_to_json(ws, { defval: null }).slice(0, 200);
        textHint = `XLSX sheet "${sheetName}" rows (first 200):\n` + JSON.stringify(json);
      }
    }

    const extracted = await analyzeWithOpenAI({
      fileUrl: file_url,
      fileName: name,
      textHint
    });

    const summary = {
      headline: extracted?.doc_type || "document",
      confidence: extracted?.confidence ?? null,
      flags: extracted?.flags ?? [],
      suggested_actions: extracted?.suggested_actions ?? []
    };

    // Send back to Hostinger webhook
    await fetch(webhook_url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-AI-Secret": process.env.AI_WEBHOOK_SECRET || ""
      },
      body: JSON.stringify({
        status: "done",
        job_id,
        upload_id,
        client_id,
        extracted,
        summary
      })
    });
  } catch (e) {
    // Best-effort error callback
    try {
      await fetch(req.body.webhook_url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-AI-Secret": process.env.AI_WEBHOOK_SECRET || ""
        },
        body: JSON.stringify({
          status: "error",
          job_id: req.body.job_id,
          upload_id: req.body.upload_id,
          client_id: req.body.client_id,
          error: String(e?.message || e)
        })
      });
    } catch {}
  }
});

app.get("/", (req, res) => res.json({ ok: true, service: "mytaxlead-ai-worker" }));

app.listen(PORT, () => console.log("AI worker listening on", PORT));