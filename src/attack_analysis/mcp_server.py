from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .agent_tools import EvidenceToolRegistry
from .benchmarks import FrozenBenchmarkStore
from .research import EvidenceBundle
from .web_search import brave_search_context


def _registry(
    *,
    bundles: dict[str, EvidenceBundle] | None = None,
    benchmarks: FrozenBenchmarkStore | None = None,
) -> EvidenceToolRegistry:
    return EvidenceToolRegistry(
        bundles or {},
        benchmarks=benchmarks,
        web_search=brave_search_context,
    )


def mcp_tool_schemas() -> list[dict[str, Any]]:
    return _registry().schemas()


def handle_mcp_request(
    message: dict[str, Any],
    *,
    registry: EvidenceToolRegistry | None = None,
) -> dict[str, Any]:
    payload = dict(message)
    request_id = payload.get("id")
    method = str(payload.get("method") or "")
    params = payload.get("params") or {}
    tools = registry or _registry()
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "attack-analysis", "version": "0.1.0"},
            },
        }
    if method in {"notifications/initialized", "notifications/cancelled"}:
        return {}
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"tools": tools.schemas()},
        }
    if method == "tools/call":
        name = str(params.get("name") or "")
        arguments = dict(params.get("arguments") or {})
        result = tools.call(name, arguments)
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]
            },
        }
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": f"method not found: {method}"},
    }


def serve_mcp_stdio(
    *,
    bundle_path: Path | None = None,
    benchmarks_path: Path | None = None,
) -> None:
    bundles: dict[str, EvidenceBundle] = {}
    if bundle_path and bundle_path.is_file():
        payload = json.loads(bundle_path.read_text(encoding="utf-8"))
        bundle = EvidenceBundle.from_dict(payload)
        bundles[bundle.attack.attack_id] = bundle
    store = FrozenBenchmarkStore.from_jsonl(benchmarks_path) if benchmarks_path else FrozenBenchmarkStore()
    registry = _registry(bundles=bundles, benchmarks=store)
    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue
        reply = handle_mcp_request(json.loads(text), registry=registry)
        if reply:
            sys.stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
            sys.stdout.flush()
