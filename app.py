import os
import json
import re
import urllib.parse
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
    if not notion or not page_id:
        return
    notion.pages.update(page_id=page_id, properties=props)

def url_for_service(req, path: str) -> str:
    """
    Prefer an explicit base (SERVICE_BASE_URL) so links are correct
    even when routes are invoked via internal/test contexts.
    """
    base = os.getenv("SERVICE_BASE_URL")
    if not base:
        base = getattr(req, "host_url", "") or ""
    if not base:
        base = "http://localhost/"  # last resort
    if not base.endswith("/"):
        base += "/"
    if path.startswith("/"):
        path = path[1:]
    return base + path


def create_notion_row(lead: dict, draft_text: str | None = None) -> str | None:
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
        "sample_link": "https://a.co/d/4GKJm8Q"
    },
    # "JINGLED": { ... }
}

ACTIVE_BOOK = os.getenv("ACTIVE_BOOK", "ASHER").upper()

# -----------------------------
# Database URL (psycopg v3 + SSL for Render)
# -----------------------------
raw_url = os.environ.get("DATABASE_URL", "") or ""
if raw_url.startswith("postgres://"):
    raw_url = "postgresql+psycopg://" + raw_url[len("postgres://"):]
elif raw_url.startswith("postgresql://"):
    raw_url = "postgresql+psycopg://" + raw_url[len("postgresql://"):]
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
# Init (base table)
# -----------------------------
@app.post("/init")
def init():
    ddl = """
    CREATE TABLE IF NOT EXISTS leads (
        id SERIAL PRIMARY KEY,
        platform TEXT NOT NULL,
        source_url TEXT NOT NULL,
        author TEXT,
        post_text TEXT,
        intent TEXT,
        topics TEXT,
        fit_score DOUBLE PRECISION,
        explanations TEXT,
        notion_page_id TEXT,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        reply_text TEXT,
        status TEXT,
        sent_at TIMESTAMPTZ,
        sent_ref TEXT,
        -- NEW: learning fields
        preset_name TEXT,
        source_sub TEXT
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))
    return jsonify({"ok": True, "message": "leads table ready"}), 200

# -----------------------------
# Migrations
# -----------------------------
@app.post("/migrate/add-topics-fit")
def migrate_add_topics_fit():
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE leads ADD COLUMN IF NOT EXISTS topics TEXT;"))
        conn.execute(text("ALTER TABLE leads ADD COLUMN IF NOT EXISTS fit_score DOUBLE PRECISION;"))
    return {"ok": True, "message": "Columns topics, fit_score ready"}, 200

@app.post("/migrate/add-explanations")
def migrate_add_explanations():
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE leads ADD COLUMN IF NOT EXISTS explanations TEXT;"))
    return {"ok": True, "message": "Column explanations ready"}, 200

@app.route("/migrate/add-send-cols", methods=["GET","POST"])
def migrate_add_send_cols():
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE leads ADD COLUMN IF NOT EXISTS reply_text TEXT;"))
        conn.execute(text("ALTER TABLE leads ADD COLUMN IF NOT EXISTS status TEXT;"))
        conn.execute(text("ALTER TABLE leads ADD COLUMN IF NOT EXISTS sent_at TIMESTAMPTZ;"))
    return {"ok": True, "message": "Columns reply_text, status, sent_at ready"}, 200

@app.route("/migrate/add-sent-ref", methods=["GET","POST"])
def migrate_add_sent_ref():
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE leads ADD COLUMN IF NOT EXISTS sent_ref TEXT;"))
    return {"ok": True, "message": "Column sent_ref ready"}, 200

@app.route("/migrate/add-learning-cols", methods=["GET","POST"])
def migrate_add_learning_cols():
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE leads ADD COLUMN IF NOT EXISTS preset_name TEXT;"))
        conn.execute(text("ALTER TABLE leads ADD COLUMN IF NOT EXISTS source_sub TEXT;"))
    return {"ok": True, "message": "Columns preset_name, source_sub ready"}, 200

# -----------------------------
# Utilities
# -----------------------------
@app.get("/count")
def count():
    with engine.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM leads")).scalar()
    return jsonify({"ok": True, "lead_count": n}), 200

def classify_intent(text_body: str) -> dict:
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

TOPIC_TAXONOMY = {
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
]

def _kebab(s: str) -> str:
    return re.sub(r"[^a-z0-9\-]+", "", re.sub(r"\s+", "-", s.strip().lower()))[:60]

def _build_analysis_prompt(post_text: str, book_profile: dict) -> str:
    return f"""You are a meticulous book-match analyst.
Return JSON with fields: topics (array, 3-10 concise kebab-case tags), fit_score (0..1), and explanations (string).

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
- Penalize mismatched audience/age-shelf; boost explicit trope/theme alignment.
- Use 0.90–1.00 for obvious direct matches; 0.70–0.89 for strong partial; 0.40–0.69 tangential; 0.10–0.39 weak.
Now analyze this post and return ONLY compact JSON.

POST: {post_text}
"""

def analyze_topics_and_fit(text_body: str, book_profile: dict) -> dict:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You produce compact JSON only."},
            {"role": "user", "content": _build_analysis_prompt(text_body, book_profile)}
        ],
        temperature=0.2
    )
    raw = resp.choices[0].message.content.strip()
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
    topics = [t for t in topics if t][:10]

    return {"topics": topics, "fit_score": score, "explanations": data.get("explanations", "")}

def generate_reply_text(row_like: dict) -> str:
    book = BOOKS.get(ACTIVE_BOOK, BOOKS["ASHER"])
    profile_note = (
        f"You're replying as the author of '{book['title']}'. "
        f"Audience: {book['audience']}. Genres: {', '.join(book['genres'])}. "
        f"Tropes: {', '.join(book['tropes'])}. Tone: {book['tone']}. "
        f"Pitch: {book['short_pitch']}. Sample link: {book['sample_link']}"
    )
    topics_display = row_like.get("topics") or "n/a"
    if isinstance(topics_display, list): topics_display = ", ".join(topics_display)
    prompt = f"""
{profile_note}

Constraints:
- 2–3 sentences, warm, witty, transparent.
- If links are allowed, include exactly one link: {book['sample_link']}

Reader post:
{row_like.get('post_text','')}
Known intent: {row_like.get('intent','CURIOUS')}
Known topics: {topics_display}
Fit score (0-1): {row_like.get('fit_score',0.0)}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"You draft brief, kind replies in the author's voice."},
            {"role":"user","content": prompt}
        ],
        temperature=0.5,
    )
    return resp.choices[0].message.content.strip()

# -----------------------------
# classify-insert
# -----------------------------
@app.post("/classify-insert")
def classify_insert():
    data = request.get_json(force=True, silent=True) or {}
    text_body = (data.get("text") or "").strip()
    if not text_body:
        return {"ok": False, "error": "Missing 'text' in JSON body"}, 400

    result = classify_intent(text_body)
    label = result["label"]
    book = BOOKS.get(ACTIVE_BOOK, BOOKS["ASHER"])
    topicfit = analyze_topics_and_fit(text_body, book)
    topics_csv = ", ".join(topicfit["topics"]) if topicfit["topics"] else None
    fit_val = float(topicfit["fit_score"])
    explain = topicfit.get("explanations") or None

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
            conn.execute(text("UPDATE leads SET reply_text=:r WHERE id=:i"), {"r": draft_text, "i": new_id})

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
        db = notion.databases.retrieve(database_id=NOTION_DB_ID)
        info["db_ok"] = True
        info["db_title"] = db.get("title", [{}])[0].get("plain_text", "")
        props = list(db.get("properties", {}).keys())
        info["props"] = props[:20]
        if "Post (snippet)" not in props:
            return {**info, "error": 'Missing required Notion property: "Post (snippet)".'}, 200
        page = notion.pages.create(
            parent={"database_id": NOTION_DB_ID},
            properties={"Post (snippet)": {"title": [{"type": "text","text": {"content": "[diag] connectivity test"}}]}}
        )
        notion.pages.update(page_id=page["id"], archived=True)
        info["create_ok"] = True
        return info, 200
    except Exception as e:
        return {**info, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[:2000]}, 200

@app.post("/analyze")
def analyze_endpoint():
    body = request.get_json(force=True, silent=True) or {}
    post_text = (body.get("post_text") or "").strip()
    if not post_text:
        return {"ok": False, "error": "Provide 'post_text' in JSON"}, 400
    book = BOOKS.get(ACTIVE_BOOK, BOOKS["ASHER"])
    result = analyze_topics_and_fit(post_text, book)
    return jsonify({"ok": True, **result}), 200

# ============================================================
# Reddit comb (RSS-based) + Presets (with fallback + fuzzy match)
# ============================================================
def _bool_arg(val: str | None, default: bool) -> bool:
    if val is None: return default
    return str(val).lower() in ("1","true","yes","on")

def _sub_from_url(url: str) -> str | None:
    m = re.search(r"reddit\.com/r/([^/]+)/", url, re.I)
    return m.group(1) if m else None

import html
import urllib.parse

def _fetch_reddit_rss(url: str):
    # Use a real UA; Reddit may return empties/429 without it.
    return feedparser.parse(url, request_headers={
        "User-Agent": os.getenv("REDDIT_USER_AGENT", "asher-agent/1.0 (+https://render.com)")
    })

def _strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", html.unescape(s or ""))

def _tokenize_query(q: str):
    """
    Returns (pos_groups, neg_terms). Supports:
      - OR groups: "queer fantasy OR lgbt fantasy"
      - minus terms: -megathread -weekly
      - quoted phrases: "royal romance"
    Any one OR-group hit qualifies; minus kills a match.
    """
    q = q.strip()
    neg = [t[1:].lower() for t in re.findall(r"\-\w[\w\-]+", q)]
    or_chunks = re.split(r"\s+OR\s+", q, flags=re.IGNORECASE)
    groups = []
    for chunk in or_chunks:
        phrases = [m.strip('"').lower() for m in re.findall(r'"([^"]+)"', chunk)]
        chunk_no_quotes = re.sub(r'"[^"]+"', " ", chunk)
        chunk_no_quotes = re.sub(r"\-\w[\w\-]+", " ", chunk_no_quotes)
        words = [w.lower() for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]+", chunk_no_quotes)]
        terms = [t for t in (phrases + words) if t]
        if terms:
            groups.append(terms)
    return groups, neg

def _fuzzy_match(text: str, pos_groups, neg_terms) -> bool:
    """
    True if:
      - none of neg_terms appear, and
      - at least one positive group has ≥1 hit (phrase or word).
    """
    t = text.lower()
    if any(n in t for n in neg_terms):
        return False
    for group in pos_groups:
        if any(term in t for term in group):
            return True
    return False

@app.route("/notion/backfill-links", methods=["POST","GET"])
def notion_backfill_links():
    """
    Rewrites Draft Link / Send Link in Notion using SERVICE_BASE_URL.
    Optional query 'id' to update just one lead: /notion/backfill-links?id=22
    """
    lead_id = request.args.get("id", type=int)
    q = "SELECT id, notion_page_id FROM leads WHERE notion_page_id IS NOT NULL"
    params = {}
    if lead_id:
        q += " AND id = :id"
        params["id"] = lead_id

    updated = 0
    base = os.getenv("SERVICE_BASE_URL") or (request.host_url if hasattr(request, "host_url") else "")
    if not base:
        return {"ok": False, "error": "SERVICE_BASE_URL not set"}, 400
    if not base.endswith("/"):
        base += "/"

    with engine.connect() as conn:
        rows = conn.execute(text(q), params).mappings().all()

    for r in rows:
        lid = int(r["id"])
        pid = r["notion_page_id"]
        props = {
            "Draft Link": notion_prop("url", f"{base}notion/fill-draft?id={lid}&force=1"),
            "Send Link":  notion_prop("url", f"{base}send?id={lid}"),
        }
        try:
            update_notion_fields(pid, props)
            updated += 1
        except Exception:
            pass

    return {"ok": True, "updated": updated}, 200

@app.route("/ingest/reddit_rss", methods=["GET", "POST"])
def ingest_reddit_rss():
    if request.method == "POST":
        body = request.get_json(force=True, silent=True) or {}
        subs = body.get("sub") or body.get("subs") or ""
        q = body.get("q") or ""
        sort = (body.get("sort") or "new").lower()
        limit = int(body.get("limit") or 10)
        autodraft_override = body.get("autodraft")
        preset_name = body.get("preset") or body.get("tag") or None
    else:
        subs = request.args.get("sub", "")
        q = request.args.get("q", "")
        sort = (request.args.get("sort") or "new").lower()
        limit = int(request.args.get("limit") or 10)
        autodraft_override = request.args.get("autodraft")
        preset_name = request.args.get("preset") or request.args.get("tag")

    subs_list = [s.strip() for s in subs.split(",") if s.strip()]
    if not subs_list or not q:
        return {"ok": False, "error": "Provide both 'sub' (comma list) and 'q' (query)."}, 400
    sort = "new" if sort not in ("new","relevance") else sort

    # autodraft policy for this run
    force_off = (str(autodraft_override).strip().lower() in ("0","false","no"))
    force_on  = (str(autodraft_override).strip().lower() in ("1","true","yes"))
    per_sub_counts, inserted_ids, total = {}, [], 0
    book = BOOKS.get(ACTIVE_BOOK, BOOKS["ASHER"])

    for sub in subs_list:
        q_enc = urllib.parse.quote_plus(q)
        rss_url = f"https://www.reddit.com/r/{sub}/search.rss?q={q_enc}&restrict_sr=1&sort={sort}"
        feed = _fetch_reddit_rss(rss_url)

        count_for_sub = 0

        # Fallback plan:
        # 1) Start with search RSS entries.
        # 2) If none, pull /new RSS and fuzzy-filter locally.
        entries = feed.entries[:limit]
        using_fallback = False
        if not entries:
            new_url = f"https://www.reddit.com/r/{sub}/new/.rss"
            new_feed = _fetch_reddit_rss(new_url)
            # scan a bit more in fallback, then filter
            entries = new_feed.entries[: max(limit, 20)]
            using_fallback = True

        # Precompute fuzzy tokens once (used only if using_fallback)
        pos_groups, neg_terms = _tokenize_query(q)

        for e in entries:
            url = getattr(e, "link", "")
            if not url:
                continue

            # skip duplicates
            with engine.connect() as conn:
                exists = conn.execute(text("SELECT 1 FROM leads WHERE source_url=:u LIMIT 1"), {"u": url}).first()
            if exists:
                continue

            author = getattr(e, "author", "reddit_user")
            title = getattr(e, "title", "") or ""
            summary_raw = getattr(e, "summary", "") or ""
            summary = _strip_html(summary_raw)
            text_body = (title.strip() + "\n\n" + summary).strip()

            # If we're in fallback mode, require fuzzy match; otherwise accept.
            if using_fallback:
                if not _fuzzy_match(text_body, pos_groups, neg_terms):
                    continue

            # classify
            result = classify_intent(text_body)
            label = result["label"]

            # topics + fit
            topicfit = analyze_topics_and_fit(text_body, book)
            topics_csv = ", ".join(topicfit["topics"]) if topicfit["topics"] else None
            fit_val = float(topicfit["fit_score"])
            explain = topicfit.get("explanations") or None

            sql = """
            INSERT INTO leads (platform, source_url, author, post_text, intent, topics, fit_score, explanations, status, preset_name, source_sub)
            VALUES (:platform, :source_url, :author, :post_text, :intent, :topics, :fit_score, :explanations, :status, :preset_name, :source_sub)
            RETURNING id;
            """
            params = {
                "platform": "reddit",
                "source_url": url,
                "author": author,
                "post_text": text_body[:5000],
                "intent": label,
                "topics": topics_csv,
                "fit_score": fit_val,
                "explanations": explain,
                "status": "new",
                "preset_name": preset_name,
                "source_sub": sub or _sub_from_url(url)
            }
            with engine.begin() as conn:
                new_id = conn.execute(text(sql), params).scalar()

            # auto-draft logic
            auto_default = (label == "BUY_READY" and fit_val >= 0.60) or (os.getenv("NOTION_AUTODRAFT","0") == "1")
            auto_draft = (False if force_off else True if force_on else auto_default)
            draft_text = None
            if auto_draft:
                draft_text = generate_reply_text({
                    "post_text": text_body,
                    "intent": label,
                    "topics": topicfit["topics"],
                    "fit_score": fit_val
                })
                with engine.begin() as conn:
                    conn.execute(text("UPDATE leads SET reply_text=:r WHERE id=:i"), {"r": draft_text, "i": new_id})

            # Notion sync
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

            count_for_sub += 1
            total += 1
            inserted_ids.append(new_id)

        per_sub_counts[sub] = count_for_sub

    return {"ok": True, "query": q, "subs": subs_list, "per_sub": per_sub_counts, "total_inserted": total, "samples": inserted_ids[:10]}, 200

# -----------------------------
# Preset definitions + runner
# -----------------------------
PRESETS = {
    "asher-hot": {
        "subs": ["books","YAlit","FantasyBooks"],
        "q": "queer YA fantasy OR (royal romance) OR (enemies to lovers) politics",
        "sort": "new",
        "limit": 10,
        "autodraft": "0"
    },
    "jingled-holiday": {
        "subs": ["books","booksuggestions","fiction"],
        "q": "holiday satire OR dark comedy family politics christmas drama",
        "sort": "new",
        "limit": 12,
        "autodraft": "0"
    }
}

def _enabled_presets():
    env = os.getenv("PRESETS_ENABLE","").strip()
    if not env:
        return list(PRESETS.keys())
    return [n.strip() for n in env.split(",") if n.strip() and n.strip() in PRESETS]

@app.route("/ingest/run-presets", methods=["GET"])
def run_presets():
    """
    Run one preset by ?name=..., or all enabled presets if no name is provided.
    If AUTODISCOVER_SUBS=1, we expand each preset's subs using recent high-fit leads.
    """
    name = request.args.get("name")
    names = [name] if name else _enabled_presets()
    ran, results = [], {}
    autodiscover = os.getenv("AUTODISCOVER_SUBS","0") == "1"
    max_auto = int(os.getenv("PRESET_MAX_AUTODISCOVER","4"))

    for n in names:
        preset = PRESETS.get(n)
        if not preset:
            results[n] = {"ok": False, "error": "unknown preset"}
            continue

        subs = list(preset["subs"])
        q = preset["q"]
        sort = preset.get("sort","new")
        limit = int(preset.get("limit",10))
        autodraft = preset.get("autodraft","0")

        # optional: expand with high-fit recent subreddits
        if autodiscover:
            with engine.connect() as conn:
                rows = conn.execute(text("""
                    SELECT DISTINCT source_sub
                    FROM leads
                    WHERE platform='reddit' AND fit_score >= 0.75 AND source_sub IS NOT NULL
                    ORDER BY created_at DESC
                    LIMIT 40
                """)).mappings().all()
            winners = [r["source_sub"] for r in rows if r["source_sub"]]
            for s in winners:
                if s not in subs and len(subs) < len(preset["subs"]) + max_auto:
                    subs.append(s)

        # call the RSS ingester for this preset
        qs = urllib.parse.urlencode({
            "sub": ",".join(subs),
            "q": q,
            "sort": sort,
            "limit": limit,
            "autodraft": autodraft,
            "preset": n
        })
        with app.test_request_context(f"/ingest/reddit_rss?{qs}", method="GET"):
            resp, status = ingest_reddit_rss()
        results[n] = {"status": status, "result": resp}
        ran.append(n)

    return {"ok": True, "ran": ran, "results": results}, 200

# -----------------------------
# Simple metrics to guide learning
# -----------------------------
@app.get("/metrics/presets")
def metrics_presets():
    q = """
    SELECT
      COALESCE(preset_name,'(none)') AS preset,
      COALESCE(source_sub,'(unknown)') AS subreddit,
      COUNT(*) AS leads,
      AVG(fit_score) AS avg_fit,
      SUM(CASE WHEN intent='BUY_READY' THEN 1 ELSE 0 END) AS buy_ready
    FROM leads
    WHERE platform='reddit'
    GROUP BY 1,2
    ORDER BY preset, avg_fit DESC NULLS LAST, leads DESC;
    """
    with engine.connect() as conn:
        rows = [dict(r) for r in conn.execute(text(q)).mappings().all()]
    return {"ok": True, "rows": rows}, 200

# -----------------------------
# Notion fill-draft
# -----------------------------
@app.get("/notion/fill-draft")
def notion_fill_draft():
    lead_id = request.args.get("id", type=int)
    force = request.args.get("force", default=0, type=int)
    if not lead_id: return jsonify({"ok": False, "error": "Missing id"}), 400

    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT id, platform, source_url, author, post_text, intent, topics, fit_score,
                   explanations, notion_page_id, created_at, reply_text
            FROM leads WHERE id = :id
        """), {"id": lead_id}).mappings().first()
    if not row: return jsonify({"ok": False, "error": "Lead not found"}), 404

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
    if not page_id: return jsonify({"ok": False, "error": "No linked Notion page_id for this lead"}), 400

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

    return jsonify({"ok": True, "lead_id": int(row["id"]), "regenerated": regenerated, "notion_page_id": page_id, "draft_preview": draft[:240]}), 200

# -----------------------------
# Send (shim or live-when-enabled)
# -----------------------------
def _is_live() -> bool: return os.getenv("SEND_LIVE", "0") == "1"

def _send_allowed(platform: str) -> bool:
    allow = (os.getenv("SEND_PLATFORMS", "reddit").split(","))
    allow = [a.strip().lower() for a in allow if a.strip()]
    return platform.lower() in allow

def _reddit_reply(source_url: str, reply_text: str) -> dict:
    if not _is_live(): return {"permalink": None, "id": None}
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
    m = re.search(r"/comments/([a-z0-9]+)/[^/]+(?:/([a-z0-9]+))?", source_url, re.I)
    if not m:
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
        parent = reddit.comment(comment_id); posted = parent.reply(reply_text)
    else:
        parent = reddit.submission(submission_id=sub_id); posted = parent.reply(reply_text)
    return {"permalink": f"https://www.reddit.com{posted.permalink}", "id": posted.id}

@app.get("/send")
def send_route():
    lead_id = request.args.get("id", type=int)
    if not lead_id: return jsonify({"ok": False, "error": "Missing id"}), 400

    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT id, platform, source_url, author, post_text, intent, topics, fit_score,
                   explanations, notion_page_id, created_at, reply_text
            FROM leads WHERE id = :id
        """), {"id": lead_id}).mappings().first()
    if not row: return jsonify({"ok": False, "error": "Lead not found"}), 404
    if not row["reply_text"] or not row["reply_text"].strip():
        return jsonify({"ok": False, "error": "No draft exists. Use /notion/fill-draft first."}), 400

    platform = (row["platform"] or "").lower()
    source_url = row["source_url"] or ""
    draft_text = row["reply_text"].strip()
    permalink = None; sent_ref = None

    if _send_allowed(platform) and platform == "reddit":
        try:
            res = _reddit_reply(source_url, draft_text)
            permalink = res.get("permalink"); sent_ref = res.get("id")
        except Exception as e:
            return jsonify({"ok": False, "error": f"Reddit send failed: {e}"}), 500

    sent_at = datetime.now(timezone.utc)
    with engine.begin() as conn:
        conn.execute(text("UPDATE leads SET status='posted', sent_at=:t, sent_ref=:r WHERE id=:i"),
                     {"t": sent_at, "r": sent_ref, "i": row["id"]})

    if row["notion_page_id"]:
        props = {"Status": notion_prop("select", {"name": "Posted"})}
        if permalink: props["Permalink"] = notion_prop("url", permalink)
        try: update_notion_fields(row["notion_page_id"], props)
        except Exception: pass

    return jsonify({"ok": True, "lead_id": int(row["id"]), "platform": platform, "source_url": source_url,
                    "permalink": permalink, "sent_at": sent_at.isoformat(), "live": _is_live()}), 200

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
