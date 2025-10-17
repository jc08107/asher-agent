# notion_sync.py
import os
from notion_client import Client

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DB_ID = os.getenv("NOTION_DB_ID")

client = Client(auth=NOTION_TOKEN)

def _ms_text(s: str):
    return [{"type": "text", "text": {"content": s[:2000]}}] if s else []

def create_row_and_optional_draft(lead: dict, draft_text: str | None = None) -> str:
    """
    lead keys expected:
      id (int), post_text (str), intent (str), fit_score (float),
      topics (list[str]), source_link (str), draft_link (str)
    Returns Notion page_id.
    """
    topics = [{"name": t} for t in (lead.get("topics") or []) if t]

    props = {
        "Post (snippet)": { "title": _ms_text((lead.get("post_text") or "")[:85]) },
        "Intent":        { "select": {"name": lead.get("intent") or "CURIOUS"} },
        "Fit":           { "number": float(lead.get("fit_score") or 0.0) },
        "Topics":        { "multi_select": topics },
        "Source Link":   { "url": lead.get("source_link") or None },
        "Draft Link":    { "url": lead.get("draft_link") or None },
        "Status":        { "select": {"name": "New"} },
        "Lead ID":       { "number": int(lead.get("id")) if lead.get("id") is not None else None },
    }

    if draft_text:
        props["Draft"] = { "rich_text": _ms_text(draft_text) }

    page = client.pages.create(
        parent={"database_id": NOTION_DB_ID},
        properties=props
    )
    return page["id"]

def update_draft(page_id: str, draft_text: str):
    client.pages.update(
        page_id=page_id,
        properties={ "Draft": { "rich_text": _ms_text(draft_text[:2000]) } }
    )
