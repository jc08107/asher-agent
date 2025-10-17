import os
import json
import re
import feedparser
from datetime import datetime, timezone
from flask import Flask, jsonify, request
from sqlalchemy import create_engine, text
from openai import OpenAI

# Notion
from notion_client import Client as NotionClient

# -----------------------------
# OpenAI client (reads OPENAI_API_KEY from env)
# -----------------------------
client = OpenAI()

# -----------------------------
# Notion client + env
# -----------------------------
NOTION_TOKEN = os.getenv("NOTION_TOKEN", "")
NOTION_DB_ID = os.getenv("NOTION_DB_ID", "")
notion = NotionClient(auth=NOTION_TOKEN) if (NOTION_TOKEN and NOTION_DB_ID) else None

def _ms_text(s: str):
    return [{"type": "text", "text": {"content": (s or "")[:2000]}}]

def notion_prop(kind, value):
    """
    kind: 'rich_text' | 'select' | 'date' | 'url' | 'number' | 'title'
    returns a Notion property payload fragment suitable for pages.update/create
    """
    if kind == "rich_text":
        return {"rich_text": _ms_text(value or "")}
    if kind == "select":
        return {"select": value}  # expects {"name": "..."}
    if kind == "date":
        return {"date": value}    # expects {"start": iso}
    if kind == "url":
        return {"url": value or None}
    if kind == "number":
        return {"number": value}
    if kind == "title":
        return {"title": _ms_text(value or "")}
    raise ValueError(f"Unknown Notion kind: {kind}")

def update_notion_fields(page_id: str, props: dict):
    """
    Shallow upsert of Notion properties; ignores if Notion is not configured.
    props must already be in Notion's shape (e.g., {"Draft": {"rich_text": [...]}, ...})
    """
    if not notion or not page_id:
        return
    notion.pages.update(page_id=page_id, properties=props)

def url_for_service(req, path: str) -> str:
    """
    Build an absolute URL back to this service for convenient 'action links' in Notion.
    """
    base = req.host_url  # includes trailing "/"
    if path.startswith("/"):
        path = path[1:]
    return base + path

def create_notion_row(lead: dict, draft_text: str | None = None) -> str | None:
    """
    Create a Notion page for a lead. Returns page_id or None if Notion is not configured.
    lead expects:
      id, post_text, intent, fit_score, topics(list[str]),
      source_link, draft_link, send_link
    """
    if not notion:
        return None

    topics = [{"name": t} for t in (lead.get("topics") or []) if t]

    props = {
        "Post (snippet)": {"title": _ms_text((lead.get("post_text") or "")[:85])},
        "Intent":        {"select": {"name": lead.get("intent") or "CURIOUS"}},
        "Fit":           {"number": float(lead.get("fit_score") or 0.0)},
        "Topics":        {"multi_select": topics},
        "Source Link":   {"url": lead.get("source_link") or None},
        "Draft Link":    {"url": lead.get("draft_link") or None},
        "Send Link":     {"url": lead.get("send_link") or None},
        "Status":        {"select": {"name": "New"}},
        "Lead ID":       {"number": int(lead["id"]) if lead.get("id") is not None else None},
    }

    if draft_text:
        props["Draft"] = {"rich_text": _ms_text(draft_text)}

    page = notion.pages.create(
        parent={"database_id": NOTION_DB_ID},
        properties=props
    )
    return page.get("id")

def update_notion_draft(page_id: str, draft_text: str):
    """Update an existing Notion page's Draft field."""
    if not notion or not page_id:
        return
    update_notion_fields(page_id, {"Draft": notion_prop("rich_text", draft_text[:2000])})

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
        return jsonify({"ok": True, "db": "connected", "notion": bool(notion)}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# -----------------------------
# Init (creates base table incl. explanations + notion_page_id)
# -----------------------------
@app.post("/init")
def init():
    """
    Create the leads table (idempotent).
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
        notion_page_id TEXT,              -- Notion page id for sync
        created_at TIMESTAMPTZ DEFAULT NOW(),
        -- new optional columns used by /notion/fill-draft and /send
        reply_text TEXT,
        status TEXT,
        sent_at TIMESTAMPTZ,
        sent_ref TEXT
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

# -----------------------------
# Migration (adds reply_text/status/sent_at)
# -----------------------------
@app.post("/migrate/add-send-cols")
def migrate_add_send_cols():
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE leads ADD COLUMN IF NOT EXISTS reply_text TEXT;"))
        conn.execute(text("ALTER TABLE leads ADD COLUMN IF NOT EXISTS status TEXT;"))
        conn.execute(text("ALTER TABLE leads ADD COLUMN IF NOT EXISTS sent_at TIMESTAMPTZ;"))
    return {"ok": True, "message": "Columns reply_text, status, sent_at ready"}, 200

# -----------------------------
# NEW Migration (adds sent_ref) — step 3
# -----------------------------
@app.route("/migrate/add-sent-ref", methods=["GET", "POST"])
def migrate_add_sent_ref():
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE leads ADD COLUMN IF NOT EXISTS sent_ref TEXT;"))
    return {"ok": True, "message": "Column sent_ref ready"}, 200

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
# Helper: generate reply text (used by /reply-draft and optional auto-draft)
# -----------------------------
def generate_reply_text(row_like: dict) -> str:
    """
    row_like must have: post_text, intent, topics, fit_score
    Uses ACTIVE_BOOK profile to produce a short reply.
    """
    book = BOOKS.get(ACTIVE_BOOK, BOOKS["ASHER"])
    profile_note = (
        f"You're replying as the author of '{book['title']}'. "
        f"Audience: {book['audience']}. Genres: {', '.join(book['genres'])}. "
        f"Tropes: {book['tropes']}. Tone: {book['tone']}. "
        f"Pitch: {book['short_pitch']}. Sample link: {book['sample_link']}"
    )

    topics_display = row_like.get("topics") or "n/a"
    if isinstance(topics_display, list):
        topics_display = ", ".join(topics_display)

    prompt = f"""
{profile_note}

Constraints:
- 2 or 3 sentences, warm, witty, transparent (no hard sell).
- Don't make claims beyond the profile.
- If links are allowed, include exactly one link: {book['sample_link']}
- If links might be banned on the platform, omit it and suggest 'peek the sample in my profile'.

Reader post:
{row_like.get('post_text','')}

Known intent: {row_like.get('intent','CURIOUS')}
Known topics: {topics_display}
Fit score (0-1): {row_like.get('fit_score',0.0)}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"You draft brief, kind, satirical-leaning replies in Evan's voice."},
            {"role":"user","content": prompt}
        ],
        temperature=0.5,
    )
    return resp.choices[0].message.content.strip()

# -----------------------------
# classify-insert: label + topics + fit (+ explanations), insert, then create/update Notion
# -----------------------------
@app.post("/classify-insert")
def classify_insert():
    """
    POST JSON: { "text": "...", "platform": "rss", "url": "https://..", "author": "..." }
    → classify with GPT → topic+fit → insert into leads → create Notion page (and optional draft)
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

    # insert lead
    sql = """
    INSERT INTO leads (platform, source_url, author, post_text, intent, topics, fit_score, explanations, status)
    VALUES (:platform, :source_url, :author, :post_text, :intent, :topics, :fit_score, :explanations, :status)
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
        "explanations": explain,
        "status": "new"
    }
    with engine.begin() as conn:
        new_id = conn.execute(text(sql), params).scalar()

    # -------- Notion Sync (direct API) --------
    # Decide whether to auto-generate a draft now
    auto_draft = (label == "BUY_READY" and fit_val >= 0.60) or (os.getenv("NOTION_AUTODRAFT","0") == "1")

    draft_text = None
    if auto_draft:
        # Generate draft in-process
        draft_text = generate_reply_text({
            "post_text": text_body,
            "intent": label,
            "topics": topicfit["topics"],
            "fit_score": fit_val
        })
        # Persist draft to DB for re-use
        with engine.begin() as conn:
            conn.execute(
                text("UPDATE leads SET reply_text=:r WHERE id=:i"),
                {"r": draft_text, "i": new_id}
            )

    # Build Notion payload (works even if draft_text is None)
    draft_link = url_for_service(request, f"/notion/fill-draft?id={new_id}&force=1")
    send_link  = url_for_service(request, f"/send?id={new_id}")

    lead_for_notion = {
        "id": new_id,
        "post_text": text_body,
        "intent": label,
        "fit_score": fit_val,
        "topics": topicfit["topics"],
        "source_link": params["source_url"],
        "draft_link": draft_link,
        "send_link": send_link,
    }

    page_id = None
    try:
        page_id = create_notion_row(lead_for_notion, draft_text=draft_text)
    except Exception:
        page_id = None

    if page_id:
        with engine.begin() as conn:
            conn.execute(text("UPDATE leads SET notion_page_id=:pid WHERE id=:id"),
                         {"pid": page_id, "id": new_id})

    return {
        "ok": True,
        "inserted_id": new_id,
        "label": label,
        "topics": topicfit["topics"],
        "fit_score": fit_val,
        "explanations": explain,
        "notion_page_id": page_id,
        "draft_link": draft_link,
        "send_link": send_link
    }, 200

@app.get("/notion/diag")
def notion_diag():
    import traceback
    info = {"env_seen": bool(notion), "db_id": os.getenv("NOTION_DB_ID", "")[:8] + "..."}
    if not notion:
        info["error"] = "Notion client not initialized. Check NOTION_TOKEN and NOTION_DB_ID."
        return info, 200
    try:
        # 1) Can we read the database?
        db = notion.databases.retrieve(database_id=NOTION_DB_ID)
        info["db_ok"] = True
        info["db_title"] = db.get("title", [{}])[0].get("plain_text", "")
        props = list(db.get("properties", {}).keys())
        info["props"] = props[:20]

        if "Post (snippet)" not in props:
            return {**info, "error": 'Missing required Notion property: "Post (snippet)". Rename your title column to exactly that.'}, 200

        page = notion.pages.create(
            parent={"database_id": NOTION_DB_ID},
            properties={"Post (snippet)": {"title": [{"type": "text","text": {"content": "[diag] connectivity test"}}]}}
        )
        notion.pages.update(page_id=page["id"], archived=True)
        info["create_ok"] = True
        return info, 200

    except Exception as e:
        return {
            **info,
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc()[:2000]
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
    Pull an RSS feed, classify entries, add topics/fit, insert rows (+ Notion sync).
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
        INSERT INTO leads (platform, source_url, author, post_text, intent, topics, fit_score, explanations, status)
        VALUES (:platform, :source_url, :author, :post_text, :intent, :topics, :fit_score, :explanations, :status)
        RETURNING id;
        """
        params = {
            "platform": "rss",
            "source_url": url,
            "author": author,
            "post_text": text_body[:5000],  # safety slice
            "intent": label,
            "topics": topics_csv,
            "fit_score": fit_val,
            "explanations": explain,
            "status": "new"
        }
        with engine.begin() as conn:
            new_id = conn.execute(text(sql), params).scalar()

        # Notion sync (respect NOTION_AUTODRAFT)
        auto_draft = (label == "BUY_READY" and fit_val >= 0.60) or (os.getenv("NOTION_AUTODRAFT","0") == "1")
        draft_text = None
        if auto_draft:
            draft_text = generate_reply_text({
                "post_text": text_body,
                "intent": label,
                "topics": topicfit["topics"],
                "fit_score": fit_val
            })
            with engine.begin() as conn:
                conn.execute(
                    text("UPDATE leads SET reply_text=:r WHERE id=:i"),
                    {"r": draft_text, "i": new_id}
                )

        draft_link = url_for_service(request, f"/notion/fill-draft?id={new_id}&force=1")
        send_link  = url_for_service(request, f"/send?id={new_id}")

        lead_for_notion = {
            "id": new_id,
            "post_text": text_body,
            "intent": label,
            "fit_score": fit_val,
            "topics": topicfit["topics"],
            "source_link": url,
            "draft_link": draft_link,
            "send_link": send_link,
        }
        try:
            page_id = create_notion_row(lead_for_notion, draft_text=draft_text)
        except Exception:
            page_id = None
        if page_id:
            with engine.begin() as conn:
                conn.execute(text("UPDATE leads SET notion_page_id=:pid WHERE id=:id"),
                             {"pid": page_id, "id": new_id})

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
        row = res.mappings().first()

    if not row:
        return {"ok": False, "error": f"Lead {lead_id} not found"}, 404

    # Prepare row-like for generator
    topics_list = [t.strip() for t in (row["topics"] or "").split(",")] if row["topics"] else []
    reply = generate_reply_text({
        "post_text": row["post_text"],
        "intent": row["intent"],
        "topics": topics_list,
        "fit_score": float(row["fit_score"] or 0.0)
    })
    return {"ok": True, "id": int(row["id"]), "reply": reply}, 200

# -----------------------------
# Notion fill-draft (clickable from Notion)
# -----------------------------
@app.get("/notion/fill-draft")
def notion_fill_draft():
    """
    GET /notion/fill-draft?id=LEAD_ID&force=1
    Generates (or reuses) a draft reply, stores it in DB, writes 'Draft' in Notion,
    sets Status='Drafted', and ensures action links exist.
    """
    lead_id = request.args.get("id", type=int)
    force = request.args.get("force", default=0, type=int)

    if not lead_id:
        return jsonify({"ok": False, "error": "Missing id"}), 400

    # fetch the lead with reply_text + notion_page_id
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT id, platform, source_url, author, post_text, intent, topics, fit_score,
                   explanations, notion_page_id, created_at, reply_text
            FROM leads
            WHERE id = :id
        """), {"id": lead_id}).mappings().first()

    if not row:
        return jsonify({"ok": False, "error": "Lead not found"}), 404

    # Decide whether to reuse or regenerate
    draft = row["reply_text"]
    regenerated = False
    if force or not (draft and draft.strip()):
        topics_list = [t.strip() for t in (row["topics"] or "").split(",")] if row["topics"] else []
        draft = generate_reply_text({
            "post_text": row["post_text"],
            "intent": row["intent"],
            "topics": topics_list,
            "fit_score": float(row["fit_score"] or 0.0)
        })
        regenerated = True
        with engine.begin() as conn:
            conn.execute(text("UPDATE leads SET reply_text=:r WHERE id=:i"), {"r": draft, "i": row["id"]})

    page_id = row["notion_page_id"]
    if not page_id:
        return jsonify({"ok": False, "error": "No linked Notion page_id for this lead"}), 400

    # Write to Notion: Draft, Status, ensure Draft/Send links
    draft_link = url_for_service(request, f"/notion/fill-draft?id={row['id']}&force=1")
    send_link  = url_for_service(request, f"/send?id={row['id']}")

    props = {
        "Draft":       notion_prop("rich_text", draft),
        "Status":      notion_prop("select", {"name": "Drafted"}),
        "Draft Link":  notion_prop("url", draft_link),
        "Send Link":   notion_prop("url", send_link),
    }
    try:
        update_notion_fields(page_id, props)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Notion update failed: {e}"}), 500

    return jsonify({
        "ok": True,
        "lead_id": int(row["id"]),
        "regenerated": regenerated,
        "notion_page_id": page_id,
        "draft_preview": draft[:240]
    }), 200

# -----------------------------
# Send helpers (feature-flagged live posting)
# -----------------------------
def _is_live() -> bool:
    return os.getenv("SEND_LIVE", "0") == "1"

def _send_allowed(platform: str) -> bool:
    allow = (os.getenv("SEND_PLATFORMS", "reddit").split(","))
    allow = [a.strip().lower() for a in allow if a.strip()]
    return platform.lower() in allow

def _reddit_reply(source_url: str, reply_text: str) -> dict:
    """
    Reply in-place on Reddit under source_url (submission or comment).
    Returns {"permalink": "...", "id": "..."}.
    Requires PRAW env vars and SEND_LIVE=1 to actually post.
    In dry-run, returns {"permalink": None, "id": None}.
    """
    if not _is_live():
        return {"permalink": None, "id": None}

    try:
        import praw
    except Exception as e:
        raise RuntimeError("praw is not installed. Add 'praw==7.8.1' to requirements.txt.") from e

    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        username=os.getenv("REDDIT_USERNAME"),
        password=os.getenv("REDDIT_PASSWORD"),
        user_agent=os.getenv("REDDIT_USER_AGENT", "asher-agent/1.0"),
    )

    # Detect whether URL points to a submission or a comment
    # Examples:
    #  - https://www.reddit.com/r/books/comments/abc123/title_slug/
    #  - https://www.reddit.com/r/books/comments/abc123/title_slug/def456/?context=3  (comment)
    m = re.search(r"/comments/([a-z0-9]+)/[^/]+(?:/([a-z0-9]+))?", source_url, re.I)
    if not m:
        # Fallback: let PRAW parse anyway
        try:
            thing = reddit.comment(url=source_url)
            posted = thing.reply(reply_text)
            return {"permalink": f"https://www.reddit.com{posted.permalink}", "id": posted.id}
        except Exception:
            thing = reddit.submission(url=source_url)
            posted = thing.reply(reply_text)
            return {"permalink": f"https://www.reddit.com{posted.permalink}", "id": posted.id}

    sub_id, comment_id = m.group(1), m.group(2)
    if comment_id:
        parent = reddit.comment(comment_id)
        posted = parent.reply(reply_text)
    else:
        parent = reddit.submission(submission_id=sub_id)
        posted = parent.reply(reply_text)

    return {"permalink": f"https://www.reddit.com{posted.permalink}", "id": posted.id}

# -----------------------------
# Send (now uses platform adapters; still safe by default)
# -----------------------------
@app.get("/send")
def send_route():
    """
    GET /send?id=LEAD_ID
    - If SEND_LIVE=0 (default), does NOT publish; marks Posted for flow testing.
    - If SEND_LIVE=1 and platform allowed, publishes and writes Permalink + sent_ref.
    """
    lead_id = request.args.get("id", type=int)
    if not lead_id:
        return jsonify({"ok": False, "error": "Missing id"}), 400

    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT id, platform, source_url, author, post_text, intent, topics, fit_score,
                   explanations, notion_page_id, created_at, reply_text
            FROM leads
            WHERE id = :id
        """), {"id": lead_id}).mappings().first()

    if not row:
        return jsonify({"ok": False, "error": "Lead not found"}), 404
    if not row["reply_text"] or not row["reply_text"].strip():
        return jsonify({"ok": False, "error": "No draft exists. Use /notion/fill-draft first."}), 400

    platform = (row["platform"] or "").lower()
    source_url = row["source_url"] or ""
    draft_text = row["reply_text"].strip()

    permalink = None
    sent_ref = None

    if _send_allowed(platform) and platform == "reddit":
        try:
            res = _reddit_reply(source_url, draft_text)
            permalink = res.get("permalink")
            sent_ref = res.get("id")
        except Exception as e:
            return jsonify({"ok": False, "error": f"Reddit send failed: {e}"}), 500
    else:
        # Unsupported or disallowed platform → no-op (still mark Posted to keep the flow)
        pass

    sent_at = datetime.now(timezone.utc)
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE leads SET status='posted', sent_at=:t, sent_ref=:r WHERE id=:i"),
            {"t": sent_at, "r": sent_ref, "i": row["id"]}
        )

    # Reflect to Notion if we have a page
    if row["notion_page_id"]:
        props = {
            "Status": notion_prop("select", {"name": "Posted"}),
        }
        # Optional: add a "Permalink" URL property in Notion and set it here
        # Make sure your Notion DB has a URL column named exactly "Permalink"
        if permalink:
            props["Permalink"] = notion_prop("url", permalink)
        try:
            update_notion_fields(row["notion_page_id"], props)
        except Exception:
            pass

    return jsonify({
        "ok": True,
        "lead_id": int(row["id"]),
        "platform": platform,
        "source_url": source_url,
        "permalink": permalink,
        "sent_at": sent_at.isoformat(),
        "live": _is_live()
    }), 200

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
