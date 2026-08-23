from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .benchmarks import FrozenBenchmarkStore
from .research import EvidenceBundle, EvaluationVerdict


ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "strict": True,
        }


class EvidenceToolRegistry:
    def __init__(
        self,
        bundles: dict[str, EvidenceBundle],
        benchmarks: FrozenBenchmarkStore | None = None,
        web_search: ToolHandler | None = None,
    ) -> None:
        self.bundles = dict(bundles)
        self.benchmarks = benchmarks or FrozenBenchmarkStore()
        self.web_search = web_search
        self.trace: list[dict[str, Any]] = []
        object_schema = {"type": "object", "additionalProperties": False}
        self.definitions = (
            ToolDefinition(
                "get_attack_evidence",
                "Return the immutable structured evidence bundle for one attack.",
                {**object_schema, "properties": {"attack_id": {"type": "string"}}, "required": ["attack_id"]},
                self._get_attack_evidence,
            ),
            ToolDefinition(
                "get_role_benchmark",
                "Return frozen benchmark records for a tactical role and optional metric.",
                {
                    **object_schema,
                    "properties": {"role": {"type": "string"}, "metric": {"type": ["string", "null"]}},
                    "required": ["role", "metric"],
                },
                self._get_role_benchmark,
            ),
            ToolDefinition(
                "resolve_player_identity",
                "Resolve a player only from frozen match, team and jersey lineup metadata; never guess.",
                {
                    **object_schema,
                    "properties": {
                        "match_id": {"type": "string"},
                        "team": {"type": "string"},
                        "jersey": {"type": "integer", "minimum": 0},
                    },
                    "required": ["match_id", "team", "jersey"],
                },
                self._resolve_player_identity,
            ),
            ToolDefinition(
                "lookup_player_snapshot",
                "Return frozen, sourced seasonal benchmark records for an already-resolved player name.",
                {**object_schema, "properties": {"player_name": {"type": "string"}}, "required": ["player_name"]},
                self._lookup_player_snapshot,
            ),
            ToolDefinition(
                "search_frozen_stats",
                "Search only the versioned frozen statistics snapshot.",
                {**object_schema, "properties": {"query": {"type": "string"}}, "required": ["query"]},
                self._search_frozen_stats,
            ),
            ToolDefinition(
                "web_search_context",
                "Search contextual web information. Results are context only and cannot support numeric verdict claims.",
                {**object_schema, "properties": {"query": {"type": "string"}}, "required": ["query"]},
                self._web_search_context,
            ),
            ToolDefinition(
                "verify_claims",
                "Verify that every claim cites evidence IDs present in the selected attack bundle.",
                {
                    **object_schema,
                    "properties": {
                        "attack_id": {"type": "string"},
                        "claims": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "text": {"type": "string"},
                                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                                },
                                "required": ["text", "evidence_ids"],
                            },
                        },
                    },
                    "required": ["attack_id", "claims"],
                },
                self._verify_claims,
            ),
        )
        self._by_name = {item.name: item for item in self.definitions}

    def schemas(self) -> list[dict[str, Any]]:
        return [item.openai_schema() for item in self.definitions]

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in self._by_name:
            raise ValueError(f"tool is not whitelisted: {name}")
        result = self._by_name[name].handler(arguments)
        self.trace.append({"tool": name, "arguments": arguments, "result": result})
        return result

    def _get_attack_evidence(self, args: dict[str, Any]) -> dict[str, Any]:
        bundle = self.bundles.get(str(args["attack_id"]))
        return {"found": bundle is not None, "bundle": bundle.to_dict() if bundle else None}

    def _get_role_benchmark(self, args: dict[str, Any]) -> dict[str, Any]:
        records = self.benchmarks.get_role_benchmark(str(args["role"]), args.get("metric"))
        return {"records": [item.to_dict() for item in records]}

    def _resolve_player_identity(self, args: dict[str, Any]) -> dict[str, Any]:
        identity = self.benchmarks.resolve_player_identity(
            str(args["match_id"]), str(args["team"]), int(args["jersey"])
        )
        return {"resolved": identity is not None, "identity": identity.to_dict() if identity else None}

    def _lookup_player_snapshot(self, args: dict[str, Any]) -> dict[str, Any]:
        records = self.benchmarks.lookup_player(str(args["player_name"]))
        return {"records": [item.to_dict() for item in records]}

    def _search_frozen_stats(self, args: dict[str, Any]) -> dict[str, Any]:
        return {"records": [item.to_dict() for item in self.benchmarks.search(str(args["query"]))]}

    def _web_search_context(self, args: dict[str, Any]) -> dict[str, Any]:
        if self.web_search is None:
            return {"enabled": False, "results": [], "limitation": "web search is not configured"}
        return self.web_search(args)

    def _verify_claims(self, args: dict[str, Any]) -> dict[str, Any]:
        bundle = self.bundles.get(str(args["attack_id"]))
        if bundle is None:
            return {"valid": False, "unsupported": ["attack_not_found"]}
        available = bundle.evidence_ids()
        unsupported = sorted(
            {
                evidence_id
                for claim in args.get("claims", ())
                for evidence_id in claim.get("evidence_ids", ())
                if evidence_id not in available
            }
        )
        missing = [claim.get("text", "") for claim in args.get("claims", ()) if not claim.get("evidence_ids")]
        return {"valid": not unsupported and not missing, "unsupported": unsupported, "claims_without_evidence": missing}


def verify_verdict(verdict: EvaluationVerdict, bundle: EvidenceBundle) -> None:
    verdict.validate(bundle.evidence_ids())
    if not bundle.downstream_ready and not verdict.abstained:
        raise ValueError("verdict must abstain when required quality gates fail")
