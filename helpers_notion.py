# helpers_notion.py

def notion_prop(kind, value):
    """
    kind: 'rich_text' | 'select' | 'date' | 'url'
    yields Notion property patch JSON
    """
    if kind == "rich_text":
        return {"rich_text": [{"type": "text", "text": {"content": value or ""}}]}
    if kind == "select":
        return {"select": value}  # expects {"name": "..."}
    if kind == "date":
        return {"date": value}    # expects {"start": iso}
    if kind == "url":
        return {"url": value}
    raise ValueError(f"Unknown notion kind {kind}")

def url_for_service(request, path):
    root = f"{request.scheme}://{request.host}/"
    return root + path
