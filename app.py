import os
from flask import Flask, jsonify
from sqlalchemy import create_engine, text

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


# Make the root friendly
@app.get("/")
def root():
    return jsonify({"message": "Jingled agent is running. Try /health"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
