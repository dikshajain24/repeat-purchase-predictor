import os
import time
import tempfile
import requests
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt

# ========= Settings =========
API_BASE = os.getenv(
    "API_BASE",
    "https://dikshajain2406-repeat-purchase-api.hf.space"  # public API base
).rstrip("/")
PREDICT = f"{API_BASE}/predict"
PREDICT_BATCH = f"{API_BASE}/predict_batch"

FEATURES = [
    "recency_days","orders","monetary","tenure_days",
    "avg_discount","return_rate","category_diversity"
]

# ========= Helpers =========
def _save_temp_csv(df: pd.DataFrame, name_hint: str = "file") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{name_hint}.csv")
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name

def make_template_csv() -> str:
    df = pd.DataFrame(columns=FEATURES)
    return _save_temp_csv(df, "template")

def make_example_csv() -> str:
    df = pd.DataFrame([
        {"recency_days": 14, "orders": 3, "monetary": 520,  "tenure_days": 365, "avg_discount": 0.10, "return_rate": 0.00, "category_diversity": 4},
        {"recency_days": 90, "orders": 1, "monetary": 75,   "tenure_days": 120, "avg_discount": 0.15, "return_rate": 0.10, "category_diversity": 1},
        {"recency_days": 5,  "orders": 7, "monetary": 1220, "tenure_days": 600, "avg_discount": 0.05, "return_rate": 0.02, "category_diversity": 6},
        {"recency_days": 30, "orders": 2, "monetary": 240,  "tenure_days": 210, "avg_discount": 0.20, "return_rate": 0.00, "category_diversity": 2},
        {"recency_days": 7,  "orders": 4, "monetary": 680,  "tenure_days": 400, "avg_discount": 0.08, "return_rate": 0.05, "category_diversity": 5},
        {"recency_days": 180,"orders": 1, "monetary": 40,   "tenure_days": 180, "avg_discount": 0.25, "return_rate": 0.00, "category_diversity": 1},
        {"recency_days": 2,  "orders": 9, "monetary": 1550, "tenure_days": 720, "avg_discount": 0.03, "return_rate": 0.01, "category_diversity": 7},
        {"recency_days": 60, "orders": 2, "monetary": 190,  "tenure_days": 300, "avg_discount": 0.12, "return_rate": 0.10, "category_diversity": 3},
        {"recency_days": 12, "orders": 5, "monetary": 820,  "tenure_days": 500, "avg_discount": 0.07, "return_rate": 0.00, "category_diversity": 6},
        {"recency_days": 45, "orders": 3, "monetary": 360,  "tenure_days": 365, "avg_discount": 0.18, "return_rate": 0.02, "category_diversity": 2},
    ])
    return _save_temp_csv(df, "example")

def _post_with_retry(url, json, timeout=30, max_retries=5):
    backoff = 0.8
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(url, json=json, timeout=timeout)
            if r.status_code == 429:
                time.sleep(backoff)
                backoff *= 1.6
                continue
            r.raise_for_status()
            return r
        except requests.exceptions.RequestException as e:
            if attempt == max_retries:
                raise gr.Error(f"API request failed after retries: {e}")
            time.sleep(backoff)
            backoff *= 1.6
    raise gr.Error("Unexpected retry loop exit")

# ========= Ping API =========
def ping_api():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=10)
        if r.status_code == 200 and r.json().get("status") == "ok":
            return f"✅ API live: {API_BASE}"
        return f"⚠️ API responded with {r.status_code}: {r.text}"
    except requests.exceptions.RequestException as e:
        return f"❌ API unreachable: {e}"

# ========= Core scoring (BATCH) =========
def score_csv(file):
    df = pd.read_csv(file.name)

    # validate schema
    for c in FEATURES:
        if c not in df.columns:
            raise gr.Error(f"❌ Missing column: {c}\nYour CSV must have exactly these headers:\n{FEATURES}")

    # Build records and cast types
    records = []
    for _, r in df.iterrows():
        rec = {
            "recency_days": float(r["recency_days"]),
            "orders": int(r["orders"]),
            "monetary": float(r["monetary"]),
            "tenure_days": float(r["tenure_days"]),
            "avg_discount": float(r["avg_discount"]),
            "return_rate": float(r["return_rate"]),
            "category_diversity": int(r["category_diversity"]),
        }
        records.append(rec)

    # Chunk to be extra safe with rate limits (e.g., 300 rows per chunk)
    CHUNK = 300
    probs_all = []
    for i in range(0, len(records), CHUNK):
        chunk = records[i:i+CHUNK]
        res = _post_with_retry(PREDICT_BATCH, json=chunk, timeout=60)
        data = res.json()
        if "probabilities" not in data:
            raise gr.Error(f"API response missing 'probabilities': {data}")
        probs_all.extend(data["probabilities"])
        if i + CHUNK < len(records):
            time.sleep(0.3)

    out = df.copy()
    out["repeat_probability"] = probs_all

    # Charts
    fig1 = plt.figure()
    plt.hist(out["repeat_probability"], bins=20, color="#f8a8cc", edgecolor="black")
    plt.title("Repeat Purchase Probability — Distribution")
    plt.xlabel("Probability"); plt.ylabel("Count")

    ranks = out["repeat_probability"].rank(method="first")
    out["decile"] = pd.qcut(ranks, 10, labels=list(range(1,11)))
    counts = out["decile"].value_counts().sort_index()

    fig2 = plt.figure()
    plt.bar(counts.index.astype(str), counts.values, color="#cbb8ff")
    plt.title("Customers per Decile (1=low, 10=high)")
    plt.xlabel("Decile"); plt.ylabel("Customers")

    download_path = _save_temp_csv(out, "scored_customers")
    return out, fig1, fig2, download_path

# ========= Lighter Pastel Theme =========
CSS = """
:root {
  --bg: #fffdfc;
  --card: #ffffff;
  --text: #1f2937;
  --muted: #6b7280;
  --pink: #f8a8cc;
  --pink-strong: #f472b6;
  --violet: #cbb8ff;
}
.gradio-container { font-family: Inter, ui-sans-serif, system-ui; background: var(--bg); color: var(--text); }
#hero h1 { color: var(--pink-strong); font-weight: 800; letter-spacing: .2px; }
#hero .tag { background: linear-gradient(90deg, var(--pink), var(--violet)); color: #3b0764; padding: 6px 12px; border-radius: 999px; display: inline-block; font-weight: 600; }
.gr-box { border-radius: 18px !important; background: var(--card); box-shadow: 0 10px 18px rgba(0,0,0,0.05); }
button { border-radius: 12px !important; font-weight: 700; background: var(--pink-strong) !important; color: white !important; border: none !important; }
small, .subtle { color: var(--muted); }
"""

# ========= UI =========
with gr.Blocks(css=CSS, title="Repeat Purchase — Beauty & Fashion") as demo:
    with gr.Row(elem_id="hero"):
        gr.Markdown(
            f"""
<div style="text-align:center; padding: 20px">
  <div class="tag">Beauty • Fashion • E-commerce</div>
  <h1>Repeat Purchase Predictor</h1>
  <p class="subtle">
    Upload a customer CSV → get repeat purchase probabilities → view charts → download scored results.
  </p>
  <small><b>API:</b> <code>{PREDICT_BATCH}</code> • <b>CSV headers:</b> {", ".join(FEATURES)}</small>
</div>
"""
        )

    with gr.Tabs():
        with gr.Tab("📂 Upload & Score"):
            with gr.Row():
                ping = gr.Button("🔌 Ping API")
                ping_out = gr.Textbox(label="API status", interactive=False)
            ping.click(fn=ping_api, outputs=ping_out)

            file_in = gr.File(file_types=[".csv"], label="Upload customer features CSV")
            go = gr.Button("✨ Score my customers")
            table = gr.Dataframe(label="📊 Scored Customers")
            plot_hist = gr.Plot(label="🔮 Probability Distribution")
            plot_deciles = gr.Plot(label="📈 Customers per Decile")
            dl_scored = gr.File(label="⬇️ Download scored CSV")

            dl_template_btn = gr.Button("Generate CSV Template")
            dl_example_btn = gr.Button("Generate Example CSV")
            dl_template = gr.File(label="⬇️ Download CSV Template")
            dl_example = gr.File(label="⬇️ Download Example CSV")

            go.click(score_csv, inputs=file_in, outputs=[table, plot_hist, plot_deciles, dl_scored])
            dl_template_btn.click(fn=make_template_csv, outputs=dl_template)
            dl_example_btn.click(fn=make_example_csv, outputs=dl_example)

        with gr.Tab("📖 User Manual"):
            gr.Markdown(f"""
### How recruiters can test this

**Option A — Quick test**
1. Click **Generate Example CSV** → Download.
2. Upload it and click **Score my customers**.
3. Review the table + charts and **Download scored CSV**.

**Option B — Use your own data**
1. Click **Generate CSV Template** to get exact headers:
```
{", ".join(FEATURES)}
```
2. Fill it with your customer features.
3. Upload and **Score my customers**.

**API (optional)**
- Batch endpoint: `POST {PREDICT_BATCH}`
- Swagger UI: {API_BASE}/docs
""")

        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
Repeat Purchase Predictor — Beauty & Fashion

**Why:** Help brands identify customers most likely to buy again.  
**How:** Explainable scoring using RFM + behavioral features.  
**Where:** API for integration + this Gradio demo.
""")

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        inbrowser=False,
        share=False
    )
