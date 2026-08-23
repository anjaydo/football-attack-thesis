from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from attack_analysis.agent_tools import EvidenceToolRegistry
from attack_analysis.attack_features import build_attack_features
from attack_analysis.benchmarks import FrozenBenchmarkStore, PlayerIdentity
from attack_analysis.llm import OpenAIResponsesClient, describe_openai_error
from attack_analysis.models import AttackSequence, Event
from attack_analysis.research import EvidenceBundle


class AgentToolTests(unittest.TestCase):
    def setUp(self) -> None:
        sequence = AttackSequence(
            attack_id="a1",
            match_id="game_1_H1",
            team="0",
            start_sec=0,
            end_sec=1,
            events=(
                Event(
                    timestamp=0,
                    event_type="pass",
                    team="0",
                    evidence_id="event:a1:0",
                ),
            ),
        )
        bundle = EvidenceBundle(
            attack=build_attack_features(sequence),
            quality_gates={"events": True, "tracking": True, "calibration": True},
            quality_metrics={},
        )
        identity = PlayerIdentity(
            match_id="game_1_H1",
            team="0",
            jersey=9,
            player_name="Verified Player",
            role="central_forward",
            source_id="lineup:game_1",
        )
        self.registry = EvidenceToolRegistry(
            {"a1": bundle}, FrozenBenchmarkStore(identities=(identity,))
        )

    def test_identity_requires_exact_registry_match(self) -> None:
        found = self.registry.call(
            "resolve_player_identity", {"match_id": "game_1_H1", "team": "0", "jersey": 9}
        )
        missing = self.registry.call(
            "resolve_player_identity", {"match_id": "game_1_H1", "team": "0", "jersey": 10}
        )
        self.assertTrue(found["resolved"])
        self.assertFalse(missing["resolved"])

    def test_claim_verifier_rejects_invented_id(self) -> None:
        result = self.registry.call(
            "verify_claims",
            {"attack_id": "a1", "claims": [{"text": "claim", "evidence_ids": ["fake"]}]},
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["unsupported"], ["fake"])

    def test_non_whitelisted_tool_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not whitelisted"):
            self.registry.call("shell", {"command": "anything"})

    def test_responses_client_executes_and_verifies_tool_calls(self) -> None:
        claim = {"text": "The attack contains a recorded pass.", "evidence_ids": ["event:a1:0"]}

        class FunctionCall:
            type = "function_call"
            name = "verify_claims"
            call_id = "call-1"
            arguments = json.dumps({"attack_id": "a1", "claims": [claim]})

            def model_dump(self, **_kwargs):
                return {
                    "type": self.type,
                    "name": self.name,
                    "call_id": self.call_id,
                    "arguments": self.arguments,
                }

        responses = iter(
            (
                SimpleNamespace(output=[FunctionCall()], output_text="", usage=SimpleNamespace()),
                SimpleNamespace(
                    output=[],
                    output_text=json.dumps(
                        {
                            "attack_id": "a1",
                            "label": "promising",
                            "score": 3,
                            "confidence": 0.7,
                            "claims": [claim],
                            "limitations": [],
                            "abstained": False,
                        }
                    ),
                    usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
                ),
            )
        )

        class FakeResponses:
            def __init__(self):
                self.requests = []

            def create(self, **request):
                self.requests.append(request)
                return next(responses)

        client = OpenAIResponsesClient.__new__(OpenAIResponsesClient)
        client.client = SimpleNamespace(responses=FakeResponses())
        client.model = "pinned-test-model"
        client.instructions = "grounded test"
        client.max_tool_rounds = 2
        client.response_language = "Vietnamese"
        verdict = client.evaluate(self.registry.bundles["a1"], self.registry)
        self.assertEqual(verdict.label, "promising")
        self.assertEqual(verdict.usage["total_tokens"], 15)
        self.assertEqual(len(client.client.responses.requests), 2)
        self.assertIn("Vietnamese", client.client.responses.requests[0]["instructions"])
        self.assertEqual(client.client.responses.requests[1]["input"][-1]["type"], "function_call_output")

    def test_openai_quota_error_is_actionable_and_sanitized(self) -> None:
        error = RuntimeError("insufficient_quota; request payload must not be displayed")
        error.status_code = 429  # type: ignore[attr-defined]
        error.code = "insufficient_quota"  # type: ignore[attr-defined]

        message = describe_openai_error(error)

        self.assertIn("accepted the API key", message)
        self.assertIn("no available quota", message)
        self.assertNotIn("request payload", message)

    def test_openai_rate_limit_is_distinct_from_quota(self) -> None:
        error = RuntimeError("rate limit reached")
        error.status_code = 429  # type: ignore[attr-defined]

        self.assertIn("rate-limited", describe_openai_error(error))

    def test_responses_client_retries_invalid_final_claim_once(self) -> None:
        verified_claim = {
            "text": "The attack contains a recorded pass.",
            "evidence_ids": ["event:a1:0"],
        }
        changed_claim = {
            "text": "The attack contains one pass.",
            "evidence_ids": ["event:not-supported"],
        }

        class FunctionCall:
            type = "function_call"
            name = "verify_claims"

            def __init__(self, call_id: str):
                self.call_id = call_id
                self.arguments = json.dumps({"attack_id": "a1", "claims": [verified_claim]})

            def model_dump(self, **_kwargs):
                return {
                    "type": self.type,
                    "name": self.name,
                    "call_id": self.call_id,
                    "arguments": self.arguments,
                }

        def final_response(claim):
            return SimpleNamespace(
                output=[],
                output_text=json.dumps(
                    {
                        "attack_id": "a1",
                        "label": "promising",
                        "score": 3,
                        "confidence": 0.7,
                        "claims": [claim],
                        "limitations": [],
                        "abstained": False,
                    }
                ),
                usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
            )

        responses = iter(
            (
                SimpleNamespace(output=[FunctionCall("call-1")], output_text="", usage=SimpleNamespace()),
                final_response(changed_claim),
                SimpleNamespace(output=[FunctionCall("call-2")], output_text="", usage=SimpleNamespace()),
                final_response(verified_claim),
            )
        )

        class FakeResponses:
            def create(self, **_request):
                return next(responses)

        client = OpenAIResponsesClient.__new__(OpenAIResponsesClient)
        client.client = SimpleNamespace(responses=FakeResponses())
        client.model = "pinned-test-model"
        client.instructions = "grounded test"
        client.max_tool_rounds = 2
        client.max_contract_retries = 1

        verdict = client.evaluate(self.registry.bundles["a1"], self.registry)

        self.assertEqual(verdict.claims[0].text, verified_claim["text"])
        self.assertEqual(verdict.usage["contract_retries"], 1)
        self.assertTrue(any(item.get("event") == "contract_retry" for item in verdict.tool_trace))

    def test_responses_client_reverifies_reworded_final_claim(self) -> None:
        verified_claim = {
            "text": "The attack contains a recorded pass.",
            "evidence_ids": ["event:a1:0"],
        }
        final_claim = {
            "text": "One pass is recorded in the attack.",
            "evidence_ids": ["event:a1:0"],
        }

        class FunctionCall:
            type = "function_call"
            name = "verify_claims"
            call_id = "call-1"
            arguments = json.dumps({"attack_id": "a1", "claims": [verified_claim]})

            def model_dump(self, **_kwargs):
                return {
                    "type": self.type,
                    "name": self.name,
                    "call_id": self.call_id,
                    "arguments": self.arguments,
                }

        responses = iter(
            (
                SimpleNamespace(output=[FunctionCall()], output_text="", usage=SimpleNamespace()),
                SimpleNamespace(
                    output=[],
                    output_text=json.dumps(
                        {
                            "attack_id": "a1",
                            "label": "promising",
                            "score": 3,
                            "confidence": 0.7,
                            "claims": [final_claim],
                            "limitations": [],
                            "abstained": False,
                        }
                    ),
                    usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
                ),
            )
        )

        class FakeResponses:
            def create(self, **_request):
                return next(responses)

        client = OpenAIResponsesClient.__new__(OpenAIResponsesClient)
        client.client = SimpleNamespace(responses=FakeResponses())
        client.model = "pinned-test-model"
        client.instructions = "grounded test"
        client.max_tool_rounds = 2
        client.max_contract_retries = 1

        verdict = client.evaluate(self.registry.bundles["a1"], self.registry)

        verification_calls = [item for item in verdict.tool_trace if item.get("tool") == "verify_claims"]
        self.assertEqual(verdict.usage["contract_retries"], 0)
        self.assertEqual(len(verification_calls), 2)
        self.assertEqual(verification_calls[-1]["arguments"]["claims"], [final_claim])


if __name__ == "__main__":
    unittest.main()
