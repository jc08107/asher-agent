import os
import json
import re
import feedparser
from datetime import datetime
from flask import Flask, jsonify, request
from sqlalchemy import create_engine, text
from openai import OpenAI

# -----------------------------
# OpenAI client (reads OPENAI_API_KEY from env)
# -----------------------------
client = OpenAI()

app = Flask(__name__)

# -----------------------------
# Book Profiles (ASHER active now; add JINGLED when ready)
# -----------------------------
BOOKS = {
    "ASHER": {
        "title": "Asher and the Prince",
        "audience": "YA (upper YA crossover friendly)",
        "genres": ["Fantasy", "Sci-Fi blend", "LGBTQ+"],
        "tropes": ["enemies to lovers", "forbidden romance", "political intrigue", "secret technology", "found family"],
        "themes": ["agency & partnership", "truth vs myth", "social change", "industrialization vs tradition"],
        "tone": "adventurous, heartfelt, thoughtful, witty",
        "comps": ["The Goblin Emperor", "Carry On", "Red, White & Royal Blue (vibes)", "Strange the Dreamer"],
        "short_pitch": (
            "A commoner with a curious mind falls for a prince in a far-future medieval world; "
            "together they must recover a mythic sword, confront a hidden empire, and decide whether the legend "
            "that unites their people is worth preserving."
        ),
        "content_notes": ["romance (queer), peril, political stakes"],
        "sample_link": "https://a.co/d/4GKJm8Q"  # TODO: replace with your real link
    },
    # "JINGLED": { ... }  # add later when you're ready
}

# Set which book is currently active for matching & reply-drafts
ACTIVE_BOOK = os.getenv("ACTIVE_BOOK", "ASHER").upper()

# -----------------------------
# Database URL (psycopg v3 + SSL for Render)
# -----------------------------
raw_url = os.environ.get("DATABASE_URL", "") or ""

# Convert legacy schemes to SQLAlchemy's psycopg v3 dialect
# postgres://... or postgresql://... -> postgresql+psycopg://...
if raw_url.startswith("postgres://"):
    raw_url = "postgresql+psycopg://" + raw_url[len("postgres://"):]
elif raw_url.startswith("postgresql://"):
    raw_url = "postgresql+psycopg://" + raw_url[len("postgresql://"):]

# Ensure SSL is required (Render Postgres expects SSL)
if raw_url and "sslmode=" not in raw_url:
    raw_url = raw_url + ("&" if "?" in raw_url else "?") + "sslmode=require"

engine = create_engine(raw_url, pool_pre_ping=True)

# -----------------------------
# Health
# -----------------------------
@app.get("/health")
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return jsonify({"ok": True, "db": "connected"}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# -----------------------------
# Init (creates base table incl. explanations)
# -----------------------------
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
        topics TEXT,                      -- comma-separated tags (optional)
        fit_score DOUBLE PRECISION,       -- 0.0 - 1.0 (optional)
        explanations TEXT,                -- why the fit was scored that way (debug)
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))
    return jsonify({"ok": True, "message": "leads table ready"}), 200

# -----------------------------
# Migration (adds topics + fit_score if you created the table earlier)
# -----------------------------
@app.post("/migrate/add-topics-fit")
def migrate_add_topics_fit():
    ddl1 = "ALTER TABLE leads ADD COLUMN IF NOT EXISTS topics TEXT;"
    ddl2 = "ALTER TABLE leads ADD COLUMN IF NOT EXISTS fit_score DOUBLE PRECISION;"
    with engine.begin() as conn:
        conn.execute(text(ddl1))
        conn.execute(text(ddl2))
    return {"ok": True, "message": "Columns topics, fit_score ready"}, 200

# -----------------------------
# Migration (adds explanations)
# -----------------------------
@app.post("/migrate/add-explanations")
def migrate_add_explanations():
    ddl = "ALTER TABLE leads ADD COLUMN IF NOT EXISTS explanations TEXT;"
    with engine.begin() as conn:
        conn.execute(text(ddl))
    return {"ok": True, "message": "Column explanations ready"}, 200

@app.post("/migrate/add-notion-page-id")
def migrate_add_notion_page_id():
    ddl = "ALTER TABLE leads ADD COLUMN IF NOT EXISTS notion_page_id TEXT;"
    with engine.begin() as conn:
        conn.execute(text(ddl))
    return {"ok": True, "message": "Column notion_page_id ready"}, 200

# -----------------------------
# Insert test row
# -----------------------------
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

# -----------------------------
# Count rows
# -----------------------------
@app.get("/count")
def count():
    with engine.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM leads")).scalar()
    return jsonify({"ok": True, "lead_count": n}), 200

# -----------------------------
# Intent classifier (BUY_READY / CURIOUS / OFF_TOPIC)
# -----------------------------
def classify_intent(text_body: str) -> dict:
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
    {text_body}

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
    try:
        data = json.loads(raw)
        label = data.get("label", "OFF_TOPIC").strip().upper()
        conf = float(data.get("confidence", 0.5))
    except Exception:
        m = re.search(r"(BUY_READY|CURIOUS|OFF_TOPIC)", raw, re.I)
        label = (m.group(1).upper() if m else "OFF_TOPIC")
        conf = 0.5
    return {"label": label, "confidence": conf}

# -----------------------------
# Enhanced Topic Taxonomy + Few-shot anchors (for stability)
# -----------------------------
TOPIC_TAXONOMY = {
    "platform-intent": ["buying-now","looking-for-recs","TBR-add","library-hold","DNF","author-follow","cover-love"],
    "age-shelf": ["YA","adult","new-adult","middle-grade"],
    "genre": ["fantasy","sci-fi","romance","thriller","mystery","speculative","historical","LGBTQ+"],
    "tropes": ["enemies-to-lovers","forbidden-love","royal-romance","found-family","political-intrigue",
               "slow-burn","queer-protagonist","family-holiday-drama","corporate-conspiracy","media-critique"],
    "vibe": ["cozy","angsty","satirical","smart","character-driven","high-stakes"]
}

FEW_SHOT = [
    {
      "post": "Looking for queer YA fantasy with politics and a royal romance—bonus if it's enemies to lovers.",
      "topics": ["YA","fantasy","romance","queer-protagonist","royal-romance","enemies-to-lovers","political-intrigue","looking-for-recs"],
      "fit_asher": 0.92, "why_asher": "Matches YA, queer royal romance, enemies-to-lovers, political intrigue.",
      "fit_jingled": 0.12, "why_jingled": "Different genre/audience."
    },
    {
      "post": "Any biting holiday satire about families melting down over politics? Dark humor welcome.",
      "topics": ["adult","satirical","family-holiday-drama","political-intrigue","media-critique","looking-for-recs"],
      "fit_asher": 0.10, "why_asher": "Not YA fantasy.",
      "fit_jingled": 0.88, "why_jingled": "Adult satire exactly on political/holiday divide."
    },
    {
      "post": "Just finished Red, White & Royal Blue. Need more royal romance with queer leads!",
      "topics": ["adult","romance","royal-romance","queer-protagonist","TBR-add"],
      "fit_asher": 0.65, "why_asher": "Queer royal romance overlap though YA fantasy vs rom-com.",
      "fit_jingled": 0.15, "why_jingled": "Not satire."
    }
]

def _kebab(s: str) -> str:
    return re.sub(r"[^a-z0-9\-]+", "", re.sub(r"\s+", "-", s.strip().lower()))[:60]

def _build_analysis_prompt(post_text: str, book_profile: dict) -> str:
    return f"""You are a meticulous book-match analyst.
Return JSON with fields: topics (array, 3-10 concise kebab-case tags), fit_score (0..1), and explanations (string).
Be decisive; avoid generic tags.

Taxonomy (examples): {json.dumps(TOPIC_TAXONOMY)}

Active book:
Title: {book_profile['title']}
Audience: {book_profile['audience']}
Genres: {', '.join(book_profile['genres'])}
Tropes: {', '.join(book_profile['tropes'])}
Themes: {', '.join(book_profile['themes'])}
Comps: {', '.join(book_profile['comps'])}
Pitch: {book_profile['short_pitch']}

Guidelines:
- Topics: pick 5–9, draw from taxonomy when relevant, plus any sharply relevant extras you invent.
- Fit scoring rubric (anchor these):
  0.90–1.00: direct, obvious match (multiple overlaps on audience/genre/tropes).
  0.70–0.89: strong partial match (2–3 major overlaps; minor mismatches ok).
  0.40–0.69: tangential (some overlap but not core).
  0.10–0.39: weak relevance.
  0.00–0.09: off-topic.
- Penalize mismatched audience/age-shelf. Boost for explicit trope & theme alignment.
- If user is BUYING/looking-now, bias fit slightly upward when content aligns.

Few-shot references:
{json.dumps(FEW_SHOT, ensure_ascii=False, indent=2)}

Now analyze this post:
POST: {post_text}

Return ONLY compact JSON.
"""

# -----------------------------
# Topic extraction + calibrated Fit scoring against ACTIVE_BOOK
# -----------------------------
def analyze_topics_and_fit(text_body: str, book_profile: dict) -> dict:
    """
    Return {"topics": [...], "fit_score": 0.0, "explanations": "..."} for the active book.
    Fit score: 0 (no fit) to 1 (strong fit) based on audience/genre/tropes/themes.
    """
    prompt = _build_analysis_prompt(text_body, book_profile)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You produce compact JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    raw = resp.choices[0].message.content.strip()
    # Parse + clamp
    try:
        data = json.loads(raw)
    except Exception:
        data = {"topics": ["unsure"], "fit_score": 0.0, "explanations": "parse_error", "raw": raw}

    try:
        score = float(data.get("fit_score", 0))
    except Exception:
        score = 0.0
    score = max(0.0, min(1.0, score))

    topics = data.get("topics") or []
    topics = [_kebab(t) for t in topics if isinstance(t, str)]
    topics = [t for t in topics if t][:10]  # 10 max

    return {
        "topics": topics,
        "fit_score": score,
        "explanations": data.get("explanations", "")
    }

# -----------------------------
# classify-insert: label + topics + fit (+ explanations), then insert
# -----------------------------
@app.post("/classify-insert")
def classify_insert():
    """
    POST JSON: { "text": "...", "platform": "rss", "url": "https://..", "author": "..." }
    → classify with GPT → topic+fit → insert into leads
    """
    data = request.get_json(force=True, silent=True) or {}
    text_body = (data.get("text") or "").strip()
    if not text_body:
        return {"ok": False, "error": "Missing 'text' in JSON body"}, 400

    # intent
    result = classify_intent(text_body)
    label = result["label"]

    # topics + fit (against active book profile)
    book = BOOKS.get(ACTIVE_BOOK, BOOKS["ASHER"])
    topicfit = analyze_topics_and_fit(text_body, book)
    topics_csv = ", ".join(topicfit["topics"]) if topicfit["topics"] else None
    fit_val = float(topicfit["fit_score"])
    explain = topicfit.get("explanations") or None

    sql = """
    INSERT INTO leads (platform, source_url, author, post_text, intent, topics, fit_score, explanations)
    VALUES (:platform, :source_url, :author, :post_text, :intent, :topics, :fit_score, :explanations)
    RETURNING id;
    """
    params = {
        "platform": data.get("platform") or "unknown",
        "source_url": data.get("url") or "n/a",
        "author": data.get("author") or "n/a",
        "post_text": text_body,
        "intent": label,
        "topics": topics_csv,
        "fit_score": fit_val,
        "explanations": explain
    }
    with engine.begin() as conn:
        new_id = conn.execute(text(sql), params).scalar()
    return {
        "ok": True,
        "inserted_id": new_id,
        "label": label,
        "topics": topicfit["topics"],
        "fit_score": fit_val,
        "explanations": explain
    }, 200

# -----------------------------
# Optional: quick analyzer endpoint for ad-hoc tests
# -----------------------------
@app.post("/analyze")
def analyze_endpoint():
    """
    POST JSON: { "post_text": "..." }
    Returns analyzer output without inserting into DB.
    """
    body = request.get_json(force=True, silent=True) or {}
    post_text = (body.get("post_text") or "").strip()
    if not post_text:
        return {"ok": False, "error": "Provide 'post_text' in JSON"}, 400
    book = BOOKS.get(ACTIVE_BOOK, BOOKS["ASHER"])
    result = analyze_topics_and_fit(post_text, book)
    return jsonify({"ok": True, **result}), 200

# -----------------------------
# ingest/rss: pull, label + topics + fit for each entry
# -----------------------------
@app.post("/ingest/rss")
def ingest_rss():
    """
    POST JSON: { "feed": "https://..." }
    Pull an RSS feed, classify entries, add topics/fit, insert rows.
    """
    body = request.get_json(force=True, silent=True) or {}
    feed_url = body.get("feed")
    if not feed_url:
        return {"ok": False, "error": "Provide 'feed' URL in JSON"}, 400

    feed = feedparser.parse(feed_url)
    if not feed.entries:
        return {"ok": False, "feed": feed_url, "inserted": 0, "error": "No entries found"}, 200

    inserted = 0
    book = BOOKS.get(ACTIVE_BOOK, BOOKS["ASHER"])
    for e in feed.entries[:20]:  # limit for now
        url = getattr(e, "link", "n/a")
        author = getattr(e, "author", "n/a")
        title = getattr(e, "title", "")
        summary = getattr(e, "summary", "")
        text_body = (title + "\n\n" + summary).strip()

        # classify intent
        result = classify_intent(text_body)
        label = result["label"]

        # topics + fit
        topicfit = analyze_topics_and_fit(text_body, book)
        topics_csv = ", ".join(topicfit["topics"]) if topicfit["topics"] else None
        fit_val = float(topicfit["fit_score"])
        explain = topicfit.get("explanations") or None

        # insert
        sql = """
        INSERT INTO leads (platform, source_url, author, post_text, intent, topics, fit_score, explanations)
        VALUES (:platform, :source_url, :author, :post_text, :intent, :topics, :fit_score, :explanations)
        """
        params = {
            "platform": "rss",
            "source_url": url,
            "author": author,
            "post_text": text_body[:5000],  # safety slice
            "intent": label,
            "topics": topics_csv,
            "fit_score": fit_val,
            "explanations": explain
        }
        with engine.begin() as conn:
            conn.execute(text(sql), params)
        inserted += 1

    return {"ok": True, "feed": feed_url, "inserted": inserted}, 200

# -----------------------------
# reply-draft: generate a 2–3 sentence reply in your voice using the active book profile
# -----------------------------
@app.post("/reply-draft")
def reply_draft():
    """
    POST JSON: { "id": <lead_id> }
    Returns a short reply tailored to the reader + active book.
    """
    data = request.get_json(force=True, silent=True) or {}
    lead_id = data.get("id")
    if not lead_id:
        return {"ok": False, "error": "Provide lead 'id' in JSON"}, 400

    # fetch the lead
    with engine.connect() as conn:
        res = conn.execute(text("""
            SELECT id, platform, source_url, author, post_text, intent, topics, fit_score
            FROM leads
            WHERE id = :id
        """), {"id": lead_id})
        row = res.fetchone()

    if not row:
        return {"ok": False, "error": f"Lead {lead_id} not found"}, 404

    book = BOOKS.get(ACTIVE_BOOK, BOOKS["ASHER"])
    profile_note = (
        f"You're replying as the author of '{book['title']}'. "
        f"Audience: {book['audience']}. Genres: {', '.join(book['genres'])}. "
        f"Tropes: {', '.join(book['tropes'])}. Tone: {book['tone']}. "
        f"Pitch: {book['short_pitch']}. Sample link: {book['sample_link']}"
    )

    prompt = f"""
{profile_note}

Constraints:
- 2 or 3 sentences, warm, witty, transparent (no hard sell).
- Don't make claims beyond the profile.
- If links are allowed, include exactly one link: {book['sample_link']}
- If links might be banned on the platform, omit it and suggest 'peek the sample in my profile'.

Reader post:
{row.post_text}

Known intent: {row.intent}
Known topics: {row.topics or 'n/a'}
Fit score (0-1): {row.fit_score if row.fit_score is not None else 0.0}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"You draft brief, kind, satirical-leaning replies in Evan's voice."},
            {"role":"user","content": prompt}
        ],
        temperature=0.5,
    )
    draft = resp.choices[0].message.content.strip()
    return {"ok": True, "id": int(row.id), "reply": draft}, 200

# -----------------------------
# Root
# -----------------------------
@app.get("/")
def root():
    return jsonify({"message": "Jingled/Asher agent is running. Try /health"}), 200

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
