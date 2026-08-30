from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Mapping


BRAVE_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
MAX_RESULTS = 3


def brave_search_api_key() -> str:
    return os.environ.get("BRAVE_SEARCH_API_KEY", "").strip()


def brave_search_configured() -> bool:
    return bool(brave_search_api_key())


def brave_search_context(args: Mapping[str, Any] | str) -> dict[str, Any]:
    """Return non-numeric web snippets. Never treat results as frozen stats."""

    query = str(args if isinstance(args, str) else args.get("query") or "").strip()
    if not query:
        return {
            "enabled": brave_search_configured(),
            "results": [],
            "limitation": "query is required; web results cannot support numeric claims",
        }
    key = brave_search_api_key()
    if not key:
        return {
            "enabled": False,
            "results": [],
            "limitation": "BRAVE_SEARCH_API_KEY is not configured",
        }
    params = urllib.parse.urlencode({"q": query, "count": MAX_RESULTS})
    request = urllib.request.Request(
        f"{BRAVE_SEARCH_ENDPOINT}?{params}",
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": key,
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=12) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as error:
        return {
            "enabled": True,
            "results": [],
            "limitation": f"Brave Search request failed: {type(error).__name__}",
        }
    web = payload.get("web") if isinstance(payload, dict) else None
    results = []
    if isinstance(web, dict):
        for index, item in enumerate(web.get("results") or []):
            if not isinstance(item, dict):
                continue
            results.append(
                {
                    "snippet_id": f"web:{index}",
                    "title": str(item.get("title") or ""),
                    "url": str(item.get("url") or ""),
                    "description": str(item.get("description") or ""),
                }
            )
            if len(results) >= MAX_RESULTS:
                break
    return {
        "enabled": True,
        "results": results,
        "limitation": "Web snippets are contextual only and cannot support numeric xG or benchmark claims.",
    }
