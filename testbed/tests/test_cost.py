"""Unit tests for litellm-based USD cost (uniform across agent engines). litellm
prices the claude-*/gpt-* ids directly and the mistral-* ids under the `mistral/`
provider prefix (results.usd_cost maps the alias). Models litellm can't price
return None, so the caller keeps any engine-reported cost."""
import json
import tempfile
import unittest
from pathlib import Path

from mlpab import results


class UsdCostTests(unittest.TestCase):
    def test_litellm_prices_claude_gpt_and_mistral(self):
        for m in ("claude-opus-4-8", "gpt-5.5",
                  "mistral-medium-3.5", "mistral-small-4", "mistral-large-3"):
            c = results.usd_cost(m, 1000, 1000)
            self.assertIsNotNone(c, m)        # mistral via the `mistral/<api-id>` mapping
            self.assertGreater(c, 0.0, m)

    def test_unpriceable_and_none_yield_none(self):
        # fable-5 isn't in litellm → None here; on the Claude engine the run keeps
        # Claude Code's own total_cost_usd (see parse_transcript test below).
        self.assertIsNone(results.usd_cost("fable-5", 100, 100))
        self.assertIsNone(results.usd_cost("totally-unknown-xyz", 100, 100))
        self.assertIsNone(results.usd_cost(None, 100, 100))

    def test_parse_transcript_fills_zero_cost_via_litellm(self):
        t = Path(tempfile.mkdtemp()) / "transcript.jsonl"
        # vibe/codex report total_cost_usd=0; litellm should fill it from tokens.
        t.write_text(json.dumps({"type": "result",
                                 "usage": {"input_tokens": 1000, "output_tokens": 1000},
                                 "num_turns": 1, "total_cost_usd": 0.0}) + "\n")
        u = results.parse_transcript_usage(t, model="mistral-medium-3.5")
        self.assertGreater(u["cost_usd"], 0.0)

    def test_parse_transcript_keeps_engine_cost_when_litellm_cant_price(self):
        t = Path(tempfile.mkdtemp()) / "transcript.jsonl"
        t.write_text(json.dumps({"type": "result",
                                 "usage": {"input_tokens": 1000, "output_tokens": 1000},
                                 "num_turns": 1, "total_cost_usd": 0.42}) + "\n")
        u = results.parse_transcript_usage(t, model="fable-5")  # unknown to litellm
        self.assertAlmostEqual(u["cost_usd"], 0.42)  # Claude-engine reported cost preserved


if __name__ == "__main__":
    unittest.main()
