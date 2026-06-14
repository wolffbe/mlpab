"""Unit tests for the Mistral Vibe agent engine: model routing (mutually
exclusive with codex/claude) and the streaming-event → transcript normalizer."""

import json
import tempfile
import unittest
from pathlib import Path

from mlpab import codex_runner, mistral_runner


class ModelRoutingTests(unittest.TestCase):
    def test_recognises_mistral(self):
        for m in ("mistral-medium-3.5", "mistral-small-4", "mistral-large-3"):
            self.assertTrue(mistral_runner.is_mistral_model(m), m)

    def test_rejects_other_engines(self):
        for m in ("claude-opus-4-8", "gpt-5.5", "gpt-5.1-codex", "fable-5"):
            self.assertFalse(mistral_runner.is_mistral_model(m), m)

    def test_mutually_exclusive_with_codex(self):
        # the dispatch in runner.py checks codex FIRST then mistral — ensure no id
        # is claimed by both engines.
        for m in ("mistral-medium-3.5", "mistral-large-3", "gpt-5.5", "claude-opus-4-8"):
            self.assertFalse(
                mistral_runner.is_mistral_model(m) and codex_runner.is_codex_model(m), m
            )


class NormalizeEventsTests(unittest.TestCase):
    def _normalize(self, events):
        d = Path(tempfile.mkdtemp())
        raw = d / "vibe_events.jsonl"
        raw.write_text("\n".join(json.dumps(e) for e in events))
        out = d / "transcript.jsonl"
        mistral_runner.normalize_events(raw, out)
        lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        result = next(l for l in lines if l["type"] == "result")
        return lines, result

    def test_vibe_messages_map_to_tool_use_and_turns(self):
        # vibe --output streaming emits role-based messages (assistant w/ tool_calls,
        # tool results); no usage in the stream.
        lines, result = self._normalize(
            [
                {
                    "role": "assistant",
                    "content": "running",
                    "tool_calls": [
                        {"function": {"name": "bash", "arguments": '{"command": "ls -la"}'}}
                    ],
                },
                {"role": "tool", "content": "file1\nfile2"},
                {"role": "assistant", "content": "done"},
            ]
        )
        tools = [l for l in lines if l["type"] == "assistant"]
        block = tools[0]["message"]["content"][0]
        self.assertEqual(block["name"], "Bash")
        self.assertEqual(block["input"]["command"], "ls -la")
        self.assertEqual(result["num_turns"], 2)  # two assistant messages
        self.assertEqual(result["usage"], {"input_tokens": 0, "output_tokens": 0})  # no meta

    def test_explicit_error_signal_flagged(self):
        # vibe's own error flag (or status) marks the tool result as an error.
        for msg in (
            {"role": "tool", "content": "boom", "is_error": True},
            {"role": "tool", "content": "boom", "status": "error"},
        ):
            lines, _ = self._normalize([msg])
            errs = [l for l in lines if l["type"] == "user"]
            self.assertTrue(errs and errs[0]["message"]["content"][0]["is_error"])

    def test_success_content_starting_with_error_not_flagged(self):
        # A successful result whose text merely starts with "error" (reading a
        # file about error handling, an echoed log line) must NOT be counted as
        # a failed command — only an explicit error signal does.
        lines, _ = self._normalize([{"role": "tool", "content": "Error: command failed"}])
        errs = [l for l in lines if l["type"] == "user"]
        self.assertTrue(errs)
        self.assertFalse(errs[0]["message"]["content"][0]["is_error"])

    def test_tokens_from_session_meta(self):
        import json as _json

        d = Path(tempfile.mkdtemp())
        (d / "vibe_events.jsonl").write_text("")
        meta = d / ".vibe" / "logs" / "session" / "s1"
        meta.mkdir(parents=True)
        (meta / "meta.json").write_text(
            _json.dumps({"session_prompt_tokens": 1234, "session_completion_tokens": 56})
        )
        out = d / "transcript.jsonl"
        mistral_runner.normalize_events(d / "vibe_events.jsonl", out)
        result = [json.loads(l) for l in out.read_text().splitlines()][-1]
        self.assertEqual(result["usage"], {"input_tokens": 1234, "output_tokens": 56})

    def test_empty_events_still_yield_result(self):
        _, result = self._normalize([])
        self.assertEqual(result["usage"], {"input_tokens": 0, "output_tokens": 0})


if __name__ == "__main__":
    unittest.main()
