import os
from flask import Flask, jsonify
from sqlalchemy import create_engine, text
import feedparser
from flask import request
from openai import OpenAI

client = OpenAI()  # uses OPENAI_API_KEY from env


app = Flask(__name__)

# --- Build a safe DB URL for psycopg v3 + SSL on Render ---
raw_url = os.environ.get("DATABASE_URL", "") or ""

# Convert legacy schemes to SQLAlchemy's psycopg v3 dialect
# e.g., postgres://...  or postgresql://...  --> postgresql+psycopg://...
if raw_url.startswith("postgres://"):
    raw_url = "postgresql+psycopg://" + raw_url[len("postgres://"):]
elif raw_url.startswith("postgresql://"):
    raw_url = "postgresql+psycopg://" + raw_url[len("postgresql://"):]

# Ensure SSL is required (Render Postgres expects SSL)
if raw_url:
    if "sslmode=" not in raw_url:
        raw_url = raw_url + ("&" if "?" in raw_url else "?") + "sslmode=require"

engine = create_engine(raw_url, pool_pre_ping=True)

# --- Simple endpoints ---

@app.get("/health")
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return jsonify({"ok": True, "db": "connected"}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/init")
def init():
    """
    Create a simple table for leads our agent will store.
    You call this once after deploy.
    """
    ddl = """
    CREATE TABLE IF NOT EXISTS leads (
        id SERIAL PRIMARY KEY,
        platform TEXT NOT NULL,           -- 'reddit', 'rss', etc.
        source_url TEXT NOT NULL,
        author TEXT,
        post_text TEXT,
        intent TEXT,                      -- 'BUY_READY', 'CURIOUS', etc.
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))
    return jsonify({"ok": True, "message": "leads table ready"}), 200


@app.post("/test-insert")
def test_insert():
    """
    Insert one fake row to confirm writes work.
    """
    sql = """
    INSERT INTO leads (platform, source_url, author, post_text, intent)
    VALUES (:platform, :source_url, :author, :post_text, :intent)
    RETURNING id;
    """
    params = {
        "platform": "demo",
        "source_url": "https://example.com/post/123",
        "author": "sample_user",
        "post_text": "I just finished White Noise and want a satire rec",
        "intent": "BUY_READY"
    }
    with engine.begin() as conn:
        new_id = conn.execute(text(sql), params).scalar()
    return jsonify({"ok": True, "inserted_id": new_id}), 200


@app.get("/count")
def count():
    """
    Count rows in leads so you can see inserts.
    """
    with engine.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM leads")).scalar()
    return jsonify({"ok": True, "lead_count": n}), 200

def classify_intent(text: str) -> dict:
    """
    Ask GPT to label text as BUY_READY / CURIOUS / OFF_TOPIC.
    Returns dict like {"label": "...", "confidence": 0.0}
    """
    prompt = f"""
    You label a short book-related post as one of:
    - BUY_READY: actively asking for the next book now
    - CURIOUS: general interest, not clearly ready to buy now
    - OFF_TOPIC: not about finding a book

    Post:
    {text}

    Reply as JSON with keys: label, confidence (0-1), rationale.
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"You are a concise intent classifier for book discovery."},
            {"role":"user","content": prompt}
        ],
        temperature=0.1,
    )
    raw = resp.choices[0].message.content.strip()
    # very simple parser: try to find label text; if JSON, great; if not, fall back
    import json, re
    try:
        data = json.loads(raw)
        label = data.get("label","OFF_TOPIC").strip().upper()
        conf = float(data.get("confidence", 0.5))
    except Exception:
        # fallback: look for BUY_READY etc.
        m = re.search(r"(BUY_READY|CURIOUS|OFF_TOPIC)", raw, re.I)
        label = (m.group(1).upper() if m else "OFF_TOPIC")
        conf = 0.5
    return {"label": label, "confidence": conf}
    

@app.post("/classify-insert")
def classify_insert():
    """
    POST JSON: { "text": "...", "platform": "rss", "url": "https://..", "author": "..." }
    → classify with GPT → insert into leads (always insert; you can filter client-side)
    """
    data = request.get_json(force=True, silent=True) or {}
    text_body = (data.get("text") or "").strip()
    if not text_body:
        return {"ok": False, "error": "Missing 'text' in JSON body"}, 400

    result = classify_intent(text_body)
    label = result["label"]

    sql = """
    INSERT INTO leads (platform, source_url, author, post_text, intent)
    VALUES (:platform, :source_url, :author, :post_text, :intent)
    RETURNING id;
    """
    params = {
        "platform": data.get("platform") or "unknown",
        "source_url": data.get("url") or "n/a",
        "author": data.get("author") or "n/a",
        "post_text": text_body,
        "intent": label
    }
    with engine.begin() as conn:
        new_id = conn.execute(text(sql), params).scalar()
    return {"ok": True, "inserted_id": new_id, "label": label, "model_raw": result}, 200


@app.post("/ingest/rss")
def ingest_rss():
    """
    POST JSON: { "feed": "https://..." }
    Pull an RSS feed, classify each entry, insert rows.
    """
    body = request.get_json(force=True, silent=True) or {}
    feed_url = body.get("feed")
    if not feed_url:
        return {"ok": False, "error": "Provide 'feed' URL in JSON"}, 400

    feed = feedparser.parse(feed_url)
    if not feed.entries:
        return {"ok": False, "feed": feed_url, "inserted": 0, "error": "No entries found"}, 200

    inserted = 0
    for e in feed.entries[:20]:  # limit for now
        url = getattr(e, "link", "n/a")
        author = getattr(e, "author", "n/a")
        title = getattr(e, "title", "")
        summary = getattr(e, "summary", "")
        text_body = (title + "\n\n" + summary).strip()

        # classify with GPT
        result = classify_intent(text_body)
        label = result["label"]

        # insert into leads table
        sql = """
        INSERT INTO leads (platform, source_url, author, post_text, intent)
        VALUES (:platform, :source_url, :author, :post_text, :intent)
        """
        params = {
            "platform": "rss",
            "source_url": url,
            "author": author,
            "post_text": text_body[:5000],  # safety slice
            "intent": label
        }
        with engine.begin() as conn:
            conn.execute(text(sql), params)
        inserted += 1

    return {"ok": True, "feed": feed_url, "inserted": inserted}, 200



# Make the root friendly
@app.get("/")
def root():
    return jsonify({"message": "Jingled agent is running. Try /health"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
