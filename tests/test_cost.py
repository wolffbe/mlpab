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
        for m in (
            "claude-opus-4-8",
            "gpt-5.5",
            "mistral-medium-3.5",
            "mistral-small-4",
            "mistral-large-3",
        ):
            c = results.usd_cost(m, 1000, 1000)
            self.assertIsNotNone(c, m)  # mistral via the `mistral/<api-id>` mapping
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
        t.write_text(
            json.dumps(
                {
                    "type": "result",
                    "usage": {"input_tokens": 1000, "output_tokens": 1000},
                    "num_turns": 1,
                    "total_cost_usd": 0.0,
                }
            )
            + "\n"
        )
        u = results.parse_transcript_usage(t, model="mistral-medium-3.5")
        self.assertGreater(u["cost_usd"], 0.0)

    def test_manual_price_wins_and_prices_unpriceable_models(self):
        price = {"input": 1.5, "output": 7.5}  # USD per 1M tokens
        # Manual price applies even when litellm can't price the model id.
        c = results.usd_cost("mistral-medium-3-5", 1_000_000, 1_000_000, price=price)
        self.assertAlmostEqual(c, 1.5 + 7.5)
        # And it wins over litellm when both could price.
        c2 = results.usd_cost("claude-opus-4-8", 2_000_000, 0, price=price)
        self.assertAlmostEqual(c2, 3.0)
        # Empty/zero manual price falls through to litellm (None for unknown ids).
        self.assertIsNone(results.usd_cost("totally-unknown-xyz", 100, 100, price={}))

    def test_parse_transcript_uses_manual_price(self):
        t = Path(tempfile.mkdtemp()) / "transcript.jsonl"
        t.write_text(
            json.dumps(
                {
                    "type": "result",
                    "usage": {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
                    "num_turns": 1,
                    "total_cost_usd": 0.0,
                }
            )
            + "\n"
        )
        u = results.parse_transcript_usage(
            t, model="mistral-medium-3-5", price={"input": 1.5, "output": 7.5}
        )
        self.assertAlmostEqual(u["cost_usd"], 9.0)

    def test_parse_transcript_keeps_engine_cost_when_litellm_cant_price(self):
        t = Path(tempfile.mkdtemp()) / "transcript.jsonl"
        t.write_text(
            json.dumps(
                {
                    "type": "result",
                    "usage": {"input_tokens": 1000, "output_tokens": 1000},
                    "num_turns": 1,
                    "total_cost_usd": 0.42,
                }
            )
            + "\n"
        )
        u = results.parse_transcript_usage(t, model="fable-5")  # unknown to litellm
        self.assertAlmostEqual(u["cost_usd"], 0.42)  # Claude-engine reported cost preserved


class CacheInclusiveCostTests(unittest.TestCase):
    """Cache-EXCLUSIVE usage reports (Claude) are billed cache inclusively:
    writes at 2x (1h) / 1.25x (5m) of the base input rate, reads at 0.1x,
    untiered writes at the 1h rate."""

    PRICE = {"input": 5.0, "output": 25.0}  # USD per 1M tokens

    def test_usd_cost_cached_arithmetic(self):
        c = results.usd_cost_cached(
            "any-model",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            cache_read_tokens=10_000_000,
            cache_write_5m_tokens=1_000_000,
            cache_write_1h_tokens=2_000_000,
            price=self.PRICE,
        )
        # 5 (in) + 20 (1h writes) + 6.25 (5m writes) + 5 (reads) + 25 (out)
        self.assertAlmostEqual(c, 61.25)

    def test_usd_cost_cached_unknown_model_is_none(self):
        self.assertIsNone(
            results.usd_cost_cached("totally-unknown-xyz", 100, 100, 100, 0, 0)
        )

    def _write_transcript(self, usage, total_cost_usd=0.0):
        t = Path(tempfile.mkdtemp()) / "transcript.jsonl"
        t.write_text(
            json.dumps(
                {
                    "type": "result",
                    "usage": usage,
                    "num_turns": 1,
                    "total_cost_usd": total_cost_usd,
                }
            )
            + "\n"
        )
        return t

    def test_parse_transcript_prices_claude_cache_fields(self):
        t = self._write_transcript(
            {
                "input_tokens": 1_000_000,
                "output_tokens": 0,
                "cache_read_input_tokens": 10_000_000,
                "cache_creation_input_tokens": 3_000_000,
                "cache_creation": {
                    "ephemeral_1h_input_tokens": 2_000_000,
                    "ephemeral_5m_input_tokens": 1_000_000,
                },
            }
        )
        u = results.parse_transcript_usage(t, model="whatever", price=self.PRICE)
        # 5 (in) + 20 (1h writes) + 6.25 (5m writes) + 5 (reads)
        self.assertAlmostEqual(u["cost_usd"], 36.25)
        # token columns stay cache exclusive
        self.assertEqual(u["input_tokens"], 1_000_000)
        self.assertEqual(u["total_tokens"], 1_000_000)

    def test_parse_transcript_bills_untiered_writes_at_1h_rate(self):
        t = self._write_transcript(
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 1_000_000,  # no tier breakdown
            }
        )
        u = results.parse_transcript_usage(t, model="whatever", price=self.PRICE)
        self.assertAlmostEqual(u["cost_usd"], 10.0)  # 1M * 5/M * 2x

    def test_parse_transcript_prefers_session_files_with_sidechains(self):
        home = Path(tempfile.mkdtemp())
        t = self._write_transcript({"input_tokens": 100, "output_tokens": 100})
        sess = home / ".claude" / "projects" / "p"
        sess.mkdir(parents=True)
        entry = {
            "type": "assistant",
            "message": {
                "id": "msg_1",
                "usage": {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
            },
        }
        sidechain = {
            "type": "assistant",
            "message": {
                "id": "msg_2",
                "usage": {"input_tokens": 0, "output_tokens": 1_000_000},
            },
        }
        # msg_1 twice (one entry per content block) → deduplicated by id
        (sess / "s.jsonl").write_text(
            "\n".join(json.dumps(e) for e in (entry, entry, sidechain)) + "\n"
        )
        u = results.parse_transcript_usage(
            t,
            model="whatever",
            price=self.PRICE,
            session_dir=home / ".claude" / "projects",
        )
        self.assertAlmostEqual(u["cost_usd"], 5.0 + 2 * 25.0)  # 1M in + 2M out
        # token columns still come from the stream-json transcript
        self.assertEqual(u["input_tokens"], 100)

    def test_parse_transcript_empty_session_dir_falls_through(self):
        home = Path(tempfile.mkdtemp())
        (home / ".claude" / "projects").mkdir(parents=True)
        t = self._write_transcript({"input_tokens": 1_000_000, "output_tokens": 1_000_000})
        u = results.parse_transcript_usage(
            t,
            model="whatever",
            price=self.PRICE,
            session_dir=home / ".claude" / "projects",
        )
        self.assertAlmostEqual(u["cost_usd"], 30.0)  # plain in+out pricing


if __name__ == "__main__":
    unittest.main()
