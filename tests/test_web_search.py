from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from attack_analysis.mcp_server import handle_mcp_request, mcp_tool_schemas
from attack_analysis.web_search import brave_search_context


class WebSearchAndMcpTests(unittest.TestCase):
    def test_brave_search_disabled_without_key(self) -> None:
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": ""}, clear=False):
            os.environ.pop("BRAVE_SEARCH_API_KEY", None)
            result = brave_search_context("football xG")
        self.assertFalse(result["enabled"])
        self.assertEqual(result["results"], [])

    def test_mcp_lists_evidence_tools(self) -> None:
        names = {item["name"] for item in mcp_tool_schemas()}
        self.assertIn("get_attack_evidence", names)
        self.assertIn("search_frozen_stats", names)
        self.assertIn("get_role_benchmark", names)
        self.assertIn("web_search_context", names)
        self.assertIn("verify_claims", names)

    def test_mcp_tools_list_request(self) -> None:
        reply = handle_mcp_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        names = {item["name"] for item in reply["result"]["tools"]}
        self.assertIn("web_search_context", names)


if __name__ == "__main__":
    unittest.main()
